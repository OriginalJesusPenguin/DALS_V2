import argparse
import sys
from os.path import join
from glob import glob
from time import perf_counter as time
from itertools import chain
from collections import defaultdict

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

import trimesh
import trimesh.graph

from skimage.measure import marching_cubes

from model.local_mod_siren import LocalModSirenDecoder
from model.loss import mesh_bl_quality_loss
from model.slice_loss import sample_points_from_intersections
from util.my_cubify import cubify
from util.data import load_meshes_in_dir
from util.metrics import point_metrics, self_intersections
from util.remesh import remesh_bk
from util import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
parser.add_argument('--checkpoint_dir')
parser.add_argument('--checkpoint_jobid')
parser.add_argument('--output_dir', default='.')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_point_samples', type=int, default=2500)
parser.add_argument('--max_iters', type=int, default=250)
parser.add_argument('--voxel_res', type=int, default=128)
parser.add_argument('--train_test_split_idx', type=int, default=0)
parser.add_argument('--sample_mode', choices=['uniform', 'slice'], default='uniform')
parser.add_argument('--remesh_at_end', action='store_true')
args = parser.parse_args()

#checkpoint_path = f'/work1/patmjen/meshfit/experiments/siren/si_{jobid}/trial_0/'

checkpoint_path = sorted(glob(join(args.checkpoint_dir, '*.ckpt')))[-1]
print('Loading checkpoint:', checkpoint_path)
checkpoint = torch.load(checkpoint_path)

hparams = checkpoint['hparams']

decoder = LocalModSirenDecoder(
    dim_in=3,
    dim_hidden=hparams['hidden_features'],
    dim_out=1,
    dim_latent=hparams['latent_features'],
    num_layers=hparams['num_hidden_layers'],
)
decoder.load_state_dict(checkpoint['decoder_state_dict'])
decoder = decoder.eval()
decoder.requires_grad_(False)
decoder.cuda()

latent_vectors = checkpoint['latent_vectors']
latent_vectors = latent_vectors.eval()
latent_vectors.requires_grad_(False)


@torch.no_grad()
def decode_to_mesh(decoder, lv, grid_size, thr=0, batch_size=None, loadbar=False, only_largest_cc=False):
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
        points_d = points[ib:ie].to(device)
        sdf_batch = decoder(points_d.unsqueeze(0), lv).cpu()
        sdf_batch[points_d.norm(dim=-1) > 1.0] = -1
        sdf[ib:ie] = sdf_batch.squeeze()

    sdf = sdf.view(grid_size, grid_size, grid_size).unsqueeze(0)
    t_forward = time() - t_forward

    t_mc = time()
    # mesh = cubify(-sdf, -thr)
    verts, faces, _, _ = marching_cubes(-sdf.numpy()[0], -thr)
    if only_largest_cc:
        tmesh = trimesh.Trimesh(verts, faces)
        cc = trimesh.graph.split(tmesh)
        largest_mesh = sorted(cc, key=lambda m: -len(m.vertices))[0]
        verts = largest_mesh.vertices
        faces = largest_mesh.faces
    mesh = Meshes(
        torch.from_numpy(verts.copy()[:, [2, 1, 0]]).unsqueeze(0).float(),
        torch.from_numpy(faces.copy()).unsqueeze(0).float()
    )
    t_mc = time() - t_mc

    return mesh, sdf, { 'points': t_points, 'forward': t_forward, 'mc': t_mc }

# '/work1/patmjen/meshfit/datasets/shapes/liver/raw/'
print('Loading data from:', args.data_path)
data = load_meshes_in_dir(args.data_path, loadbar=False)
train_data = data[:args.train_test_split_idx]
val_data = data[args.train_test_split_idx:]

def decode_from_pointset_local(eval_decoder, lv0, fit_points, lv_grid_size, max_iter=300, lr=1e-2, lr_step=100, loadbar=False):
    assert lv0.shape[1] % (lv_grid_size ** 3) == 0
    latent_features = lv0.shape[1] // (lv_grid_size ** 3)
    #lv_grid = torch.zeros((1, latent_features) + (lv_grid_size,) * 3, device=lv0.device)
    lv_grid = lv0.view(1, latent_features, lv_grid_size, lv_grid_size, lv_grid_size).clone()
    lv_grid.requires_grad = True

    optimizer = torch.optim.Adam([lv_grid], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)

    decoder.eval()
    decoder.requires_grad = False

    losses = []
    for i in trange(max_iter) if loadbar else range(max_iter):

        pred_sdf = eval_decoder(lv_grid, fit_points.unsqueeze(0))

        loss = torch.abs(pred_sdf).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

    lv_grid = lv_grid.detach()
    return lv_grid, losses

def eval_reconstruction(mesh_pred, mesh_true, num_points=100000):
    points_pred = sample_points_from_meshes(mesh_pred, num_points)
    points_true = sample_points_from_meshes(mesh_true, num_points)

    metrics = point_metrics(points_pred, points_true, [0.02, 0.04])
    metrics['Chamfer'] = chamfer_distance(points_pred, points_true)[0].cpu()

    return metrics


lv0 = latent_vectors.weight.data.mean(dim=0, keepdims=True)

eval_decoder = lambda lv, points: decoder(points, lv)

def print_metrics_table(train_metrics, val_metrics, stream):
    print('Training', file=stream)
    for key, val in train_metrics.items():
        print(f"{key:20}: {val.mean():6.2f} ± {val.std():6.2f}", file=stream)

    print('', file=stream)
    print('Validation', file=stream)
    for key, val in val_metrics.items():
        print(f"{key:20}: {val.mean():6.2f} ± {val.std():6.2f}", file=stream)

seed_everything(1337)

print('Evaluating validation data')
all_val_meshes = []
all_val_metrics = []
for i, true_mesh in enumerate(val_data):
    print(f'{i}')
    true_mesh = true_mesh.cuda()
    if args.sample_mode == 'uniform':
        fit_points = sample_points_from_meshes(true_mesh, args.num_point_samples).cuda()[0]
    elif args.sample_mode == 'slice':
        plane_normals = torch.eye(3, device=true_mesh.device)
        plane_dists = torch.zeros(3, device=true_mesh.device) + 0.1
        fit_points = sample_points_from_intersections(
            true_mesh.verts_packed(),
            true_mesh.faces_packed(),
            plane_normals,
            plane_dists,
            args.num_point_samples,
            return_point_weights=False,
            return_normals=False,
        )
    t0 = time()
    lv = decode_from_pointset_local(
        eval_decoder,
        lv0,
        fit_points,
        hparams['lv_grid_size'],
        max_iter=args.max_iters,
        lr=args.lr,
        lr_step=100
    )[0]
    t1 = time()
    pred_mesh, sdf, prof_times = decode_to_mesh(
        decoder,
        lv,
        args.voxel_res,
        thr=0.0,
        batch_size=128**3,
        only_largest_cc=True
    )
    t2 = time()


    pred_mesh.scale_verts_(2/args.voxel_res)
    pred_mesh.offset_verts_(torch.tensor([-1, -1, -1]))

    if args.remesh_at_end:
        pred_mesh = remesh_bk(pred_mesh)

    all_val_meshes.append(pred_mesh.detach().cpu().clone())
    metrics = eval_reconstruction(pred_mesh.cuda(), true_mesh.cuda(), 100000)
    metrics['BL quality'] = 1.0 - mesh_bl_quality_loss(pred_mesh)
    metrics['TimeOptim'] = t1 - t0
    metrics['TimeMC'] = t2 - t1
    metrics['TimeTotal'] = t2 - t0
    all_val_metrics.append(metrics)

val_metrics = defaultdict(list)
for m in all_val_metrics:
    for key, val in m.items():
        val_metrics[key].append(val.item() if isinstance(val, torch.Tensor) else val)

for k in val_metrics.keys():
    val_metrics[k] = np.asarray(val_metrics[k])

val_metrics['Chamfer'] *= 1e4


print_metrics_table(val_metrics, val_metrics, sys.stdout)

seed_everything(1337)

print('Evaluating training data')
all_train_metrics = []
all_train_meshes = []
for i, true_mesh in enumerate(train_data):
    print(f'{i}')
    true_mesh = true_mesh.cuda()
    if args.sample_mode == 'uniform':
        fit_points = sample_points_from_meshes(true_mesh, args.num_point_samples).cuda()[0]
    elif args.sample_mode == 'slice':
        plane_normals = torch.eye(3, device=true_mesh.device)
        plane_dists = torch.zeros(3, device=true_mesh.device) + 0.1
        fit_points = sample_points_from_intersections(
            true_mesh.verts_packed(),
            true_mesh.faces_packed(),
            plane_normals,
            plane_dists,
            args.num_point_samples,
            return_point_weights=False,
            return_normals=False,
        )
    t0 = time()
    lv = decode_from_pointset_local(
        eval_decoder,
        lv0,
        fit_points,
        hparams['lv_grid_size'],
        max_iter=args.max_iters,
        lr=args.lr,
        lr_step=100
    )[0]
    t1 = time()
    pred_mesh, sdf, prof_times = decode_to_mesh(
        decoder,
        lv,
        args.voxel_res,
        thr=0.0,
        batch_size=2*128**3,
        only_largest_cc=True
    )
    t2 = time()


    pred_mesh.scale_verts_(2/args.voxel_res)
    pred_mesh.offset_verts_(torch.tensor([-1, -1, -1]))

    if args.remesh_at_end:
        pred_mesh = remesh_bk(pred_mesh)

    all_train_meshes.append(pred_mesh.detach().cpu().clone())
    metrics = eval_reconstruction(pred_mesh.cuda(), true_mesh.cuda(), 100000)
    metrics['BL quality'] = 1.0 - mesh_bl_quality_loss(pred_mesh)
    metrics['TimeOptim'] = t1 - t0
    metrics['TimeMC'] = t2 - t1
    metrics['TimeTotal'] = t2 - t0
    all_train_metrics.append(metrics)

print('Done')

train_metrics = defaultdict(list)
for m in all_train_metrics:
    for key, val in m.items():
        train_metrics[key].append(val.item() if isinstance(val, torch.Tensor) else val)

for k in train_metrics.keys():
    train_metrics[k] = np.asarray(train_metrics[k])

train_metrics['Chamfer'] *= 1e4

print_metrics_table(train_metrics, val_metrics, sys.stdout)
with open(join(args.output_dir, 'metrics.txt'), 'w') as f:
    print_metrics_table(train_metrics, val_metrics, f)

with open(join(args.output_dir, 'args.txt'), 'w') as f:
    print(args, file=f)

out_fname = join(args.output_dir, 'inference_results.pt')
print('Writing results to:', out_fname)
torch.save({
    'train_meshes': all_train_meshes,
    'val_meshes': all_val_meshes,
    'train_metrics': train_metrics,
    'val_metrics': val_metrics,
    'args': args,
}, out_fname)

print('All done. Exiting')
