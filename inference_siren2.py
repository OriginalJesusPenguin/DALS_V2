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

from model.siren.modules import SingleBVPNet
from model.siren_decoder_2 import SirenDecoder
from model.loss import mesh_bl_quality_loss
from util.my_cubify import cubify
from util.data import load_meshes_in_dir
from util.metrics import point_metrics, self_intersections
try:
    from util.remesh import remesh_bk
    REMESH_AVAILABLE = True
except ImportError:
    print("Warning: Remesh functionality not available - missing libremesh.so")
    REMESH_AVAILABLE = False
    def remesh_bk(mesh, iters=5):
        return mesh
from util import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
parser.add_argument('--checkpoint_dir')
parser.add_argument('--checkpoint_jobid')
parser.add_argument('--model_type', choices=['siren', 'mehta', 'mehta_grid'])
parser.add_argument('--output_dir', default='.')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_point_samples', type=int, default=2500)
parser.add_argument('--max_iters', type=int, default=250)
parser.add_argument('--lv_grid_size', type=int, default=1)
parser.add_argument('--voxel_res', type=int, default=128)
parser.add_argument('--remesh_at_end', action='store_true')
parser.add_argument('--train_test_split_idx', type=int, default=0)
args = parser.parse_args()

# Ensure output directory exists
import os
os.makedirs(args.output_dir, exist_ok=True)

#checkpoint_path = f'/work1/patmjen/meshfit/experiments/siren/si_{jobid}/trial_0/'

# checkpoint_path = sorted(glob(join(args.checkpoint_dir, '*.ckpt')))[-1]
# checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/SirenDecoderTrainer_2025-10-03_16-33.ckpt'
# checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/SirenDecoderTrainer_2025-10-03_16-35.ckpt'
# checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/SirenDecoderTrainer_2025-10-03_16-36.ckpt'
checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/SirenDecoderTrainer_2025-10-03_16-37.ckpt'
# checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/SirenDecoderTrainer_2025-10-03_16-39.ckpt'
# checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/SirenDecoderTrainer_2025-10-03_17-10.ckpt'

print('Loading checkpoint:', checkpoint_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(checkpoint_path, map_location=device)

hparams = checkpoint['hparams']

# Check what model type was actually saved
if 'siren.layers.0.weight' in checkpoint['decoder_state_dict']:
    # This is a SirenDecoder model
    print("Detected SirenDecoder model in checkpoint")
    decoder = SirenDecoder(
        dim_in=3,
        dim_hidden=hparams['hidden_features'],
        dim_out=1,
        dim_latent=hparams['latent_features'],
        num_layers=hparams['num_hidden_layers'],
    )
    # Override model_type to match the actual saved model
    args.model_type = 'mehta'
elif args.model_type == 'siren':
    # Handle missing 'type' key with default value
    decoder_type = hparams.get('type', 'sine')
    decoder = SingleBVPNet(
        type=decoder_type,
        in_features=hparams['latent_features'] + 3,
        mode=hparams.get('mode', 'mlp'),
        hidden_features=hparams['hidden_features'],
        num_hidden_layers=hparams['num_hidden_layers'],
    )
elif args.model_type == 'mehta' or args.model_type == 'mehta_grid':
    decoder = SirenDecoder(
        dim_in=3,
        dim_hidden=hparams['hidden_features'],
        dim_out=1,
        dim_latent=hparams['latent_features'],
        num_layers=hparams['num_hidden_layers'],
    )
decoder.load_state_dict(checkpoint['decoder_state_dict'])
decoder = decoder.eval()
decoder.requires_grad_(False)
decoder.to(device)

latent_vectors = checkpoint['latent_vectors']
latent_vectors = latent_vectors.eval()
latent_vectors.requires_grad_(False)
latent_vectors.to(device)

def eval_decoder_1(decoder, lv, points):
    decoder_input = torch.cat([lv.expand(len(points), -1), points], dim=1)
    return decoder({ 'coords': decoder_input })['model_out']


def eval_decoder_2(decoder, lv, points):
    return decoder(points, lv)


def eval_decoder_2_grid(decoder, lv_grid, points):
    # Fix grid sampling dimensions: lv_grid should be [B, C, D, H, W] and points_grid should be [B, 1, 1, N, 3]
    # where N is the number of points
    points_grid = points.view((1, 1, 1, -1, 3))  # [1, 1, 1, N, 3]
    # Normalize points to [-1, 1] range for grid_sample
    points_grid = 2.0 * points_grid - 1.0
    lv = F.grid_sample(lv_grid, points_grid, mode='bilinear', padding_mode='border', align_corners=False)
    lv = lv.squeeze().T  # [N, C]
    return eval_decoder_2(decoder, lv, points)


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
        if args.model_type == 'siren':
            sdf_batch = eval_decoder_1(decoder, lv, points_d).cpu()
        elif args.model_type == 'mehta':
            sdf_batch = eval_decoder_2(decoder, lv, points_d).cpu()
        elif args.model_type == 'mehta_grid':
            sdf_batch = eval_decoder_2_grid(decoder, lv, points_d).cpu()
        sdf_batch[points_d.norm(dim=-1) > 1.0] = 1
        sdf[ib:ie] = sdf_batch.squeeze()

    sdf = sdf.view(grid_size, grid_size, grid_size).unsqueeze(0)
    t_forward = time() - t_forward

    t_mc = time()
    try:
        sdf_np = sdf.numpy()[0]
        # Adjust threshold to be within the data range
        sdf_min, sdf_max = sdf_np.min(), sdf_np.max()
        if -thr < sdf_min or -thr > sdf_max:
            # Use a threshold within the data range
            thr = -sdf_min + 0.1 * (sdf_max - sdf_min)
        verts, faces, _, _ = marching_cubes(-sdf_np, -thr)
        if only_largest_cc:
            tmesh = trimesh.Trimesh(verts, faces)
            cc = trimesh.graph.split(tmesh)
            if len(cc) == 0:
                raise RuntimeError('No connected components after marching cubes')
            largest_mesh = sorted(cc, key=lambda m: -len(m.vertices))[0]
            verts = largest_mesh.vertices
            faces = largest_mesh.faces
        mesh = Meshes(
            torch.from_numpy(verts.copy()[:, [2, 1, 0]]).unsqueeze(0).float(),
            torch.from_numpy(faces.copy()).unsqueeze(0).float()
        )
        t_mc = time() - t_mc
        return mesh, sdf, { 'points': t_points, 'forward': t_forward, 'mc': t_mc }
    except Exception as e:
        print(f'[WARN] decode_to_mesh failed: {e}')
        return None, None, { 'points': t_points, 'forward': t_forward, 'mc': 0.0 }

# '/work1/patmjen/meshfit/datasets/shapes/liver/raw/'
print('Loading data from:', args.data_path)
data = load_meshes_in_dir(args.data_path, loadbar=False)
train_data = data[:args.train_test_split_idx]
val_data = data[args.train_test_split_idx:]

# Derive filenames for each mesh to save alongside latent vectors
import glob as _glob
mesh_filenames = sorted(_glob.glob(os.path.join(args.data_path, '*.obj')))
mesh_filenames = [os.path.basename(fname) for fname in mesh_filenames]

def decode_from_pointset(eval_decoder, lv0, fit_points, max_iter=300, lr=1e-2, lr_step=100, loadbar=False):
    lv = lv0.clone()
    lv.requires_grad = True

    optimizer = torch.optim.Adam([lv], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)

    decoder.eval()
    decoder.requires_grad = False

    losses = []
    for i in trange(max_iter) if loadbar else range(max_iter):
        pred_sdf = eval_decoder(lv, fit_points)

        loss = torch.abs(pred_sdf).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

    lv = lv.detach()
    return lv, losses

def decode_from_pointset_local(eval_decoder, lv0, fit_points, lv_grid_size, max_iter=300, lr=1e-2, lr_step=100, loadbar=False):
    lv_grid = torch.zeros((1, lv0.shape[1]) + (lv_grid_size,) * 3, device=lv0.device)
    lv_grid += lv0.view(1, lv0.shape[1], 1, 1, 1)
    lv_grid.requires_grad = True

    # Fix grid sampling dimensions
    fit_points_grid = fit_points.view((1, 1, 1, -1, 3))  # [1, 1, 1, N, 3]
    # Normalize points to [-1, 1] range for grid_sample
    fit_points_grid = 2.0 * fit_points_grid - 1.0

    optimizer = torch.optim.Adam([lv_grid], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)

    decoder.eval()
    decoder.requires_grad = False

    losses = []
    for i in trange(max_iter) if loadbar else range(max_iter):
        lv = F.grid_sample(lv_grid, fit_points_grid, mode='bilinear', padding_mode='border', align_corners=False)
        lv = lv.squeeze().T  # [N, C]

        pred_sdf = eval_decoder(lv, fit_points)

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
    metrics['ChamferL2 x 10000'] = chamfer_distance(points_pred, points_true)[0].cpu()

    return metrics

seed_everything(1337)

lv0 = latent_vectors.weight.data.mean(dim=0, keepdims=True)

if args.model_type == 'siren':
    eval_decoder = lambda lv, points: eval_decoder_1(decoder, lv, points)
elif args.model_type == 'mehta':
    eval_decoder = lambda lv, points: eval_decoder_2(decoder, lv, points)
elif args.model_type == 'mehta_grid':
    eval_decoder = lambda lv, points: eval_decoder_2_grid(decoder, lv, points)

print('Evaluating training data')
all_train_metrics = []
all_train_meshes = []
all_latent_vectors = []
all_filenames = []
for i, true_mesh in enumerate(train_data):
    try:
        print(f'{i}')
        fit_points = sample_points_from_meshes(true_mesh, args.num_point_samples).to(device)[0]
        t0 = time()
        if args.model_type == 'mehta_grid':
            lv = decode_from_pointset_local(
                eval_decoder,
                lv0,
                fit_points,
                args.lv_grid_size,
                max_iter=args.max_iters,
                lr=args.lr,
                lr_step=100
            )[0]
        else:
            lv = decode_from_pointset(
                eval_decoder,
                lv0,
                fit_points,
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
        if pred_mesh is None:
            raise RuntimeError('Pred mesh generation failed')
        t2 = time()

        pred_mesh.scale_verts_(2/args.voxel_res)
        pred_mesh.offset_verts_(torch.tensor([-1, -1, -1]))

        pred_mesh = remesh_bk(pred_mesh, iters=5)

        all_train_meshes.append(pred_mesh.detach().cpu().clone())
        metrics = eval_reconstruction(pred_mesh.to(device), true_mesh.to(device), 100000)
        print(metrics['ChamferL2 x 10000'])
        metrics['BL quality'] = 1.0 - mesh_bl_quality_loss(pred_mesh)
        #metrics['No. ints.'] = 100 * float(self_intersections(pred_mesh.cpu())[0][0]) / len(pred_mesh.faces_packed())
        metrics['TimeOptim'] = t1 - t0
        metrics['TimeMC'] = t2 - t1
        metrics['TimeTotal'] = t2 - t0
        all_train_metrics.append(metrics)

        # Store filename and optimized latent representation
        if args.train_test_split_idx != 0:
            filename_idx = i
        else:
            filename_idx = i
        if filename_idx < len(mesh_filenames):
            all_filenames.append(mesh_filenames[filename_idx])
        else:
            all_filenames.append(f'train_{i}.obj')
        all_latent_vectors.append(lv.detach().cpu().clone())
    except Exception as e:
        print(f'[WARN] Skipping training sample {i}: {e}')

seed_everything(1337)

print('Evaluating validation data')
all_val_meshes = []
all_val_metrics = []
for i, true_mesh in enumerate(val_data):
    try:
        print(f'{i}')
        fit_points = sample_points_from_meshes(true_mesh, args.num_point_samples).to(device)[0]
        t0 = time()
        if args.model_type == 'mehta_grid':
            lv = decode_from_pointset_local(
                eval_decoder,
                lv0,
                fit_points,
                args.lv_grid_size,
                max_iter=args.max_iters,
                lr=args.lr,
                lr_step=100
            )[0]
        else:
            lv = decode_from_pointset(
                eval_decoder,
                lv0,
                fit_points,
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
        if pred_mesh is None:
            raise RuntimeError('Pred mesh generation failed')
        t2 = time()

        pred_mesh.scale_verts_(2/args.voxel_res)
        pred_mesh.offset_verts_(torch.tensor([-1, -1, -1]))

        pred_mesh = remesh_bk(pred_mesh, iters=5)

        all_val_meshes.append(pred_mesh.detach().cpu().clone())
        metrics = eval_reconstruction(pred_mesh.to(device), true_mesh.to(device), 100000)
        print(metrics['ChamferL2 x 10000'])
        metrics['BL quality'] = 1.0 - mesh_bl_quality_loss(pred_mesh)
        #metrics['No. ints.'] = 100 * float(self_intersections(pred_mesh.cpu())[0][0]) / len(pred_mesh.faces_packed())
        metrics['TimeOptim'] = t1 - t0
        metrics['TimeMC'] = t2 - t1
        metrics['TimeTotal'] = t2 - t0
        all_val_metrics.append(metrics)

        # Store filename and optimized latent representation for validation split
        if args.train_test_split_idx != 0:
            filename_idx = args.train_test_split_idx + i
        else:
            filename_idx = i
        if filename_idx < len(mesh_filenames):
            all_filenames.append(mesh_filenames[filename_idx])
        else:
            all_filenames.append(f'val_{i}.obj')
        all_latent_vectors.append(lv.detach().cpu().clone())
    except Exception as e:
        print(f'[WARN] Skipping validation sample {i}: {e}')

print('Done')

train_metrics = defaultdict(list)
for m in all_train_metrics:
    for key, val in m.items():
        train_metrics[key].append(val.item() if isinstance(val, torch.Tensor) else val)

val_metrics = defaultdict(list)
for m in all_val_metrics:
    for key, val in m.items():
        val_metrics[key].append(val.item() if isinstance(val, torch.Tensor) else val)

for k in list(train_metrics.keys()):
    train_metrics[k] = np.asarray(train_metrics[k]) if len(train_metrics[k]) > 0 else np.asarray([])

for k in list(val_metrics.keys()):
    val_metrics[k] = np.asarray(val_metrics[k]) if len(val_metrics[k]) > 0 else np.asarray([])

# Scale Chamfer distance to match mesh_decoder format (L2 * 10000)
if args.train_test_split_idx != 0:
    train_metrics['ChamferL2 x 10000'] *= 1e1
val_metrics['ChamferL2 x 10000'] *= 1e1

def print_metrics_table(train_metrics, val_metrics, stream):
    print('Training', file=stream)
    print('-' * 50, file=stream)
    for key, val in train_metrics.items():
        print(f"{key:20}: {val.mean():8.4f} ± {val.std():8.4f}", file=stream)

    print('', file=stream)
    print('Validation', file=stream)
    print('-' * 50, file=stream)
    for key, val in val_metrics.items():
        print(f"{key:20}: {val.mean():8.4f} ± {val.std():8.4f}", file=stream)

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

# Save latent vectors and filenames with automatic suffix if file exists (match mesh_decoder behavior)
base_latent_fname = join(args.output_dir, 'latent_vectors.pt')
latent_fname = base_latent_fname
latent_counter = 1
while os.path.exists(latent_fname):
    latent_fname = join(args.output_dir, f'latent_vectors_{latent_counter}.pt')
    latent_counter += 1

print('Writing latent vectors to:', latent_fname)
torch.save({
    'filenames': all_filenames,
    'latent_vectors': all_latent_vectors,
    'args': args,
}, latent_fname)

# Print summary of saved files (align with mesh_decoder)
print(f'\nFiles saved:')
print(f'  - Inference results: {os.path.basename(out_fname)}')
print(f'  - Latent vectors: {os.path.basename(latent_fname)}')
if latent_counter > 1:
    print(f'  - Note: Files were saved with suffixes to avoid overwriting existing files')
