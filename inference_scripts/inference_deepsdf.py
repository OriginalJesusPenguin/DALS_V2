import sys
from pathlib import Path

if Path(sys.path[0]).parts[-1] == 'inference_scripts':
    sys.path[0] = str(Path(sys.path[0]).parent.absolute())

import argparse
from os.path import join
from glob import glob
from time import perf_counter as time
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.measure import marching_cubes

from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes

from util import seed_everything
from util.data import load_meshes_in_dir
from util.metrics import point_metrics, self_intersections
from util.remesh import remesh_bk
from model.loss import mesh_bl_quality_loss
from model.deep_sdf_decoder import DeepSdfDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
parser.add_argument('--checkpoint_dir')
parser.add_argument('--checkpoint_jobid')
parser.add_argument('--output_dir', default='.')
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--num_point_samples', type=int, default=2500)
parser.add_argument('--max_iters', type=int, default=1000)
parser.add_argument('--voxel_res', type=int, default=128)
parser.add_argument('--remesh_at_end', action='store_true')
parser.add_argument('--train_test_split_idx', type=int, default=0)
args = parser.parse_args()

print(args)

#jobid = 12683871
#checkpoint_path = f'/work1/patmjen/meshfit/experiments/deep_sdf/sd_{jobid}/trial_0/'

checkpoint_path = sorted(glob(join(args.checkpoint_dir, '*.ckpt')))[-1]
print('Loading checkpoint', checkpoint_path)
checkpoint = torch.load(checkpoint_path)

hparams = checkpoint['hparams']

decoder = DeepSdfDecoder(
    hparams['latent_size'],
    hparams['dims'],
    dropout=hparams['dropout'],
    dropout_prob=hparams['dropout_prob'],
    norm_layers=hparams['norm_layers'],
    latent_in=hparams['latent_in'],
    weight_norm=hparams['weight_norm'],
    xyz_in_all=hparams['xyz_in_all'],
    use_tanh=hparams['use_tanh'],
    latent_dropout=hparams['latent_dropout'],
)
decoder.load_state_dict(checkpoint['decoder_state_dict'])
decoder = decoder.eval()
decoder.requires_grad_(False)
decoder.cuda()

latent_vectors = checkpoint['latent_vectors']
latent_vectors = latent_vectors.eval()
latent_vectors.requires_grad_(False)

@torch.no_grad()
def eval_decoder(decoder, lv, points):
    decoder_input = torch.cat([lv.expand(len(points), -1), points], dim=1)
    return decoder(decoder_input)


@torch.no_grad()
def decode_to_mesh(decoder, lv, grid_size, thr=0, batch_size=None, loadbar=False):
    if lv.ndim == 1:
        lv = lv.unsqueeze(0)
    if batch_size is None:
        points_device = None
        batch_size = grid_size ** 3
    else:
        points_device = lv.device

    device = lv.device

    t_points = time()
    grid_axis = torch.linspace(-1.0, 1.0, grid_size, device=points_device)
    gz, gy, gx = torch.meshgrid(grid_axis, grid_axis, grid_axis)
    points = torch.stack([gx.ravel(), gy.ravel(), gz.ravel()]).T
    t_points = time() - t_points

    t_forward = time()
    sdf = torch.empty(len(points))
    range_fn = trange if loadbar else range
    for ib in range_fn(0, len(points), batch_size):
        ie = min(len(points), ib + batch_size)
        sdf_batch = eval_decoder(decoder, lv, points[ib:ie].to(device)).cpu()
        sdf[ib:ie] = sdf_batch.squeeze()

    sdf = sdf.view(grid_size, grid_size, grid_size).unsqueeze(0)
    t_forward = time() - t_forward

    t_mc = time()
    # mesh = cubify(-sdf, -thr)
    verts, faces, _, _ = marching_cubes(-sdf.numpy()[0], -thr)
    mesh = Meshes(torch.from_numpy(verts.copy()[:, [2, 1, 0]]).unsqueeze(0), torch.from_numpy(faces.copy()).unsqueeze(0))
    t_mc = time() - t_mc

    return mesh, sdf, { 'points': t_points, 'forward': t_forward, 'mc': t_mc }

print('Loading data from:', args.data_path)
meshes = load_meshes_in_dir(args.data_path, loadbar=False)

num_points = args.num_point_samples
device = 'cuda'
lr = args.lr
max_iter = args.max_iters

cf = 10000

print('Starting inference')

all_metrics_0 = []
all_meshes_0 = []
all_metrics_001 = []
all_meshes_001 = []
for i, true_mesh in enumerate(meshes):
    print(f'{i}')
    true_mesh = true_mesh.to(device)
    points = sample_points_from_meshes(true_mesh, num_points)[0]
    sdf_true = torch.zeros(num_points, device=device)

    lv = torch.mean(latent_vectors.weight.data, dim=0)
    lv.requires_grad = True

    optimizer = torch.optim.Adam([lv], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max_iter // 4, gamma=0.5)

    l1_loss = nn.L1Loss()

    decoder = decoder.to(device)
    decoder = decoder.eval()
    decoder.requires_grad = False

    losses = []

    t0 = time()
    for i in range(max_iter):
        optimizer.zero_grad()

        decoder_input = torch.cat([lv.expand(len(points), -1), points], dim=1)
        sdf_pred = decoder(decoder_input)

        loss = l1_loss(sdf_pred.squeeze(), sdf_true)
        # loss += 1e-4 * torch.mean(lv ** 2)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.cpu().item())
    t1 = time()

    res = args.voxel_res
    got_0_level_set = True
    try:
        mesh, sdf, prof_times = decode_to_mesh(decoder, lv, res, thr=0.0, batch_size=2*128**3)

        mesh.scale_verts_(2/res)
        mesh.offset_verts_(torch.tensor([-1, -1, -1]))

        if args.remesh_at_end:
            mesh = remesh_bk(mesh)

        all_meshes_0.append(mesh)

        true_point_samples = sample_points_from_meshes(true_mesh.cuda(), 100000)
        pred_point_samples = sample_points_from_meshes(mesh.cuda(), 100000)

        metrics = point_metrics(true_point_samples, pred_point_samples, [0.02, 0.04])
        metrics[f'ChamferL2 x {cf}'] = chamfer_distance(true_point_samples, pred_point_samples)[0] * cf
        metrics['BL quality'] = 1.0 - mesh_bl_quality_loss(mesh)
        metrics['No. ints.'] = 100 * float(self_intersections(mesh.cpu())[0][0]) / len(mesh.faces_packed())
        metrics['Search'] = t1 - t0
        for k, v in prof_times.items():
            metrics[f'Decode/{k}'] = v

        all_metrics_0.append(metrics)
    except:
        got_0_level_set = False


    mesh, sdf, prof_times = decode_to_mesh(decoder, lv, res, thr=0.001, batch_size=2*128**3)

    mesh.scale_verts_(2/res)
    mesh.offset_verts_(torch.tensor([-1, -1, -1]))

    if args.remesh_at_end:
        mesh = remesh_bk(mesh)

    all_meshes_001.append(mesh)

    true_point_samples = sample_points_from_meshes(true_mesh.cuda(), 100000)
    pred_point_samples = sample_points_from_meshes(mesh.cuda(), 100000)

    metrics = point_metrics(true_point_samples, pred_point_samples, [0.02, 0.04])
    metrics[f'ChamferL2 x {cf}'] = chamfer_distance(true_point_samples, pred_point_samples)[0] * cf
    metrics['BL quality'] = 1.0 - mesh_bl_quality_loss(mesh)
    metrics['No. ints.'] = 100 * float(self_intersections(mesh.cpu())[0][0]) / len(mesh.faces_packed())
    metrics['Search'] = t1 - t0
    for k, v in prof_times.items():
        metrics[f'Decode/{k}'] = v

    all_metrics_001.append(metrics)
    if not got_0_level_set:
        all_metrics_0.append(metrics)

print('Done')

metrics = defaultdict(list)
for m in all_metrics_0:
    for k, v, in m.items():
        metrics[k].append(v)

metrics = { k: torch.tensor(v) for k, v in metrics.items() }

def print_metrics_table(metrics, train_test_split_idx, stream=sys.stdout):
    if train_test_split_idx != 0:
        print('Training', file=stream)
        print('-' * 50, file=stream)
        for k, v in metrics.items():
            vals = v[:train_test_split_idx]
            print(f'{k:20}: {vals.mean():8.4f} ± {vals.std():8.4f}', file=stream)
        print('')

    print('Validation', file=stream)
    print('-' * 50, file=stream)
    for k, v in metrics.items():
        vals = v[train_test_split_idx:]
        print(f'{k:20}: {vals.mean():8.4f} ± {vals.std():8.4f}', file=stream)
    print('', file=stream)

print_metrics_table(metrics, args.train_test_split_idx)
with open(join(args.output_dir, 'metrics.txt'), 'w') as f:
    print_metrics_table(metrics, args.train_test_split_idx, f)

with open(join(args.output_dir, 'args.txt'), 'w')  as f:
    print(args, file=f)

out_fname = join(args.output_dir, 'inference_results.pt')
print('Writing results to:', out_fname)
torch.save({
    'all_metrics_0': all_metrics_0,
    'all_metrics_001': all_metrics_001,
    'all_meshes_0': all_meshes_0,
    'all_meshes_001': all_meshes_001,
    'args': args,
}, out_fname)

print('All done. Exiting')
