import sys
import random
import datetime
import os
from collections import defaultdict
from time import time
from glob import glob
from os.path import join
import argparse

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch3d
import pytorch3d.io
import pytorch3d.utils
from pytorch3d.structures import (
    Meshes,
    Pointclouds,
    join_meshes_as_batch,
)
from pytorch3d.ops import (
    sample_points_from_meshes,
    knn_points,
    GraphConv,
    SubdivideMeshes,
)
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.ops import laplacian, sample_farthest_points
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
)

from augment.point_wolf import augment_meshes
from util.data import load_meshes_in_dir, sample_meshes

from model.mesh_decoder import (
    MeshDecoder,
    seed_everything,
)

# Import metrics with error handling
try:
    from util.metrics import point_metrics, self_intersections
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import metrics module: {e}")
    METRICS_AVAILABLE = False
    
    # Create fallback functions
    def point_metrics(*args, **kwargs):
        return {}
    
    def self_intersections(meshes):
        return torch.zeros(len(meshes)), torch.zeros(1)
try:
    from util.remesh import remesh_template_from_deformed, remesh_bk
except ImportError:
    # Remesh functionality not available
    def remesh_template_from_deformed(*args, **kwargs):
        raise NotImplementedError("Remesh functionality not available - missing libremesh.so")
    def remesh_bk(*args, **kwargs):
        raise NotImplementedError("Remesh functionality not available - missing libremesh.so")

from model.loss import mesh_bl_quality_loss, mesh_edge_loss_highdim, mesh_laplacian_loss_highdim

subdivide = SubdivideMeshes()

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
parser.add_argument('--checkpoint_dir')
parser.add_argument('--checkpoint_jobid')
parser.add_argument('--output_dir', default='.')
parser.add_argument('--latent_mode', choices=['global', 'local'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_point_samples', type=int, default=2500)
parser.add_argument('--point_sample_mode', choices=['uniform', 'fps'],
                    default='uniform')
parser.add_argument('--max_iters', type=int, default=250)
parser.add_argument('--template_subdiv', type=int, default=4)
parser.add_argument('--remesh_with_forward_at_end', action='store_true')
parser.add_argument('--remesh_at_end', action='store_true')
parser.add_argument('--remesh_at', type=float, default=[], nargs='*')
parser.add_argument('--train_test_split_idx', type=int, default=0)
parser.add_argument('--weight_bl_quality_loss', type=float, default=1e-3)
parser.add_argument('--weight_edge_length_loss', type=float, default=1e2)
args = parser.parse_args()

print(args)

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Load model checkpoint
#jobid = 12880693
#checkpoint_path = f'/work1/patmjen/meshfit/experiments/mesh_decoder/md_{jobid}/trial_0/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# checkpoint_path = sorted(glob(join(args.checkpoint_dir, '*MeshDecoder*.ckpt')))[-1]
# checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/MeshDecoderTrainer_2025-10-03_15-58.ckpt'
checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/scripts/rami_scripts/MeshDecoderTrainer_2025-10-16_10-50.ckpt'
# checkpoint_path = '/home/ralbe/DALS/mesh_autodecoder/LocalMeshDecoderTrainer_2025-10-03_16-29.ckpt'
print('Loading checkpoint:',  checkpoint_path)

checkpoint = torch.load(checkpoint_path, map_location=device)

hparams = checkpoint['hparams']

decoder = MeshDecoder(
    hparams['latent_features'],
    hparams['steps'],
    hparams['hidden_features'],
    hparams['subdivide'],
    mode=hparams['decoder_mode'],
    norm=hparams['normalization'][0],
)
decoder.load_state_dict(checkpoint['decoder_state_dict'])
decoder.eval()


latent_vectors = checkpoint['latent_vectors']
latent_vectors.eval()
template = checkpoint['template']

# Load and augment data

#data_path = '/work1/patmjen/meshfit/datasets/shapes/liver/raw/'
#data_path = '/work1/patmjen/meshfit/datasets/shapes/spleen/raw/'
#data_path = '/work1/patmjen/meshfit/datasets/shapes/ShapeNetV2/planes/'
print('Loading data from:', args.data_path)
meshes = load_meshes_in_dir(args.data_path)
print('Found', len(meshes), 'meshes')

# Get filenames for each mesh
import glob
mesh_filenames = sorted(glob.glob(os.path.join(args.data_path, '*.obj')))
mesh_filenames = [os.path.basename(fname) for fname in mesh_filenames]  # Get just the filename, not full path
print('Found', len(mesh_filenames), 'filenames')

#num_point_samples = 2500
#max_iters = 250
#lr = 1e-3

remesh_at = []
for r in args.remesh_at:
    if r < 1.0:
        remesh_at.append(r * args.max_iters)
    else:
        remesh_at.append(r)

cf = 10000

all_metrics = []

all_pred_meshes = []
all_latent_vectors = []
all_filenames = []
print('Running inference')
if args.latent_mode == 'global':
    for i, true_mesh in enumerate(tqdm(meshes, desc="Inference", ncols=100)):
        true_mesh = true_mesh.to(device)
        true_points = sample_points_from_meshes(true_mesh, args.num_point_samples)

        lv = latent_vectors.weight.data.mean(dim=0).clone().to(device).unsqueeze(0)
        lv.requires_grad_(True)

        optim = torch.optim.Adam([lv], lr=args.lr)

        #search_template = subdivide(template.to(device))
        search_template = pytorch3d.utils.ico_sphere(args.template_subdiv,
                                                     device=device)
        search_template.scale_verts_(0.1)

        decoder.eval()
        decoder.requires_grad_(False)
        decoder.to(device)

        min_loss = np.inf
        no_improvement_iters = 0
        best_iter = -1
        best_lv = lv

        t0 = time()
        for it in range(args.max_iters):
            optim.zero_grad()

            pred_mesh = decoder(search_template, lv)[-1]

            pred_points = sample_points_from_meshes(pred_mesh,
                                                    args.num_point_samples)
            loss = chamfer_distance(pred_points, true_points)[0]

            loss.backward()

            if loss < min_loss:
                min_loss = loss
                no_improvement_iters = 0
                best_iter = it
                best_lv = lv
            else:
                no_improvement_iters += 1

            if no_improvement_iters > 10:
                break

            optim.step()

            if it in remesh_at:
                with torch.no_grad():
                    search_template = remesh_template_from_deformed(
                        pred_mesh,
                        search_template,
                    )

        t1 = time()

        if args.remesh_with_forward_at_end:
            with torch.no_grad():
                search_template = remesh_template_from_deformed(
                    pred_mesh,
                    search_template,
                    ratio=0.6
                )
        t2 = time()

        with torch.no_grad():
            pred_mesh = decoder(search_template, best_lv)[-1]

            if args.remesh_at_end:
                v0, v1 = pred_mesh.verts_packed()[pred_mesh.edges_packed()].unbind(1)
                h = torch.norm(v1 - v0, dim=1).mean().cpu()
                # Use keyword args to avoid passing target_length_ratio positionally
                pred_mesh = remesh_bk(pred_mesh, target_length=h, iters=5)
        t3 = time()

        all_pred_meshes.append(pred_mesh.clone().cpu())
        
        # Store filename and optimized latent vector
        all_filenames.append(mesh_filenames[i])
        all_latent_vectors.append(best_lv.clone().cpu())

        true_point_samples = sample_points_from_meshes(true_mesh, 100000)
        pred_point_samples = sample_points_from_meshes(pred_mesh, 100000)

        metrics = point_metrics(true_point_samples, pred_point_samples, [0.01, 0.02])
        metrics[f'ChamferL2 x {cf}'] = chamfer_distance(true_point_samples, pred_point_samples)[0] * cf
        metrics['BL quality'] = 1.0 - mesh_bl_quality_loss(pred_mesh)
        if METRICS_AVAILABLE:
            metrics['No. ints.'] = 100 * float(self_intersections(pred_mesh.cpu())[0][0]) / len(pred_mesh.faces_packed())
        else:
            metrics['No. ints.'] = 0.0  # Fallback when C++ extension not available
        metrics['Search'] = t1 - t0
        metrics['Remesh'] = t2 - t1
        metrics['Decode'] = t3 - t2
        metrics['Total'] = t3 - t0

        all_metrics.append(metrics)
elif args.latent_mode == 'local':
    
    for i, true_mesh in enumerate(tqdm(meshes, desc="Local latent optimization")):
        true_mesh = true_mesh.to(device)
        if args.point_sample_mode == 'uniform':
            true_points = sample_points_from_meshes(true_mesh,
                                                    args.num_point_samples)
        else: #args.point_sample_mode == 'fps'
            num_init_samples = max(10000, args.num_point_samples * 100)
            all_true_points = sample_points_from_meshes(true_mesh,
                                                        num_init_samples)
            true_points = sample_farthest_points(
                all_true_points,
                K=args.num_point_samples,
            )[0]

        #search_template = subdivide(template.to(device))
        search_template = pytorch3d.utils.ico_sphere(args.template_subdiv,
                                                     device=device)
        search_template.scale_verts_(0.1)

        lv = latent_vectors.weight.data.mean(dim=0).clone().to(device).unsqueeze(0)
        lv = lv.repeat(len(search_template.verts_packed()), 1)
        lv.requires_grad_(True)

        num_verts = len(search_template.verts_packed())
        #L = laplacian(search_template.verts_packed(), search_template.edges_packed()).to_dense()
        #I = torch.eye(num_verts, device=L.device)
        #M = I - 1e-2 * L
        #M_inv = torch.linalg.inv(M)
        #M_inv2 = M_inv @ M_inv
        #M_inv4 = M_inv2 @ M_inv2

        decoder.eval()
        decoder.requires_grad_(False)
        decoder.to(device)

        optim = torch.optim.Adam([lv], lr=args.lr)
        losses = []

        min_loss = np.inf
        no_improvement_iters = 0
        best_iter = -1
        best_lv = lv

        m1 = torch.zeros_like(lv)
        m2 = torch.zeros_like(lv)
        beta1 = 0.9
        beta2 = 0.999

        t0 = time()
        for it in range(args.max_iters):
            optim.zero_grad()

            lv.requires_grad_(True)
            pred_mesh = decoder(search_template, lv, expand_lv=False)[-1]

            pred_points = sample_points_from_meshes(pred_mesh,
                                                    args.num_point_samples)
            loss = chamfer_distance(pred_points, true_points)[0]
            loss += args.weight_bl_quality_loss * mesh_bl_quality_loss(pred_mesh)
            # loss += args.weight_edge_length_loss * mesh_edge_loss_highdim(pred_mesh, lv)
            loss += args.weight_edge_length_loss * mesh_laplacian_loss_highdim(pred_mesh, lv)
            # loss += 1e-2 * mesh_laplacian_smoothing(pred_mesh)
            # loss += 1e-2 * torch.sum(torch.norm(L.mm(pred_mesh.verts_packed()), dim=1) / num_verts)

            loss.backward()

            if loss < 1.05 * min_loss:
                min_loss = loss
                no_improvement_iters = 0
                best_iter = it
                best_lv = lv
            else:
                no_improvement_iters += 1

            if no_improvement_iters > 10:
                pass
                # break

            optim.step()
            #with torch.no_grad():
            #    k = it + 1
            #    g = lv.grad
            #    m1 = beta1 * m1 + (1 - beta1) * g
            #    m2 = beta2 * m2 + (1 - beta2) * (g ** 2)
            #    lv -= args.lr / ((1 - beta1 ** k) * torch.sqrt(m2.max() / (1 - beta2 ** k))) * m1
            #lv.grad.zero_()

            if it in remesh_at:
                with torch.no_grad():
                    search_template, vert_features = remesh_template_from_deformed(
                        pred_mesh,
                        search_template,
                        vert_features=[lv, best_lv, m1, m2],
                    )
                    lv, best_lv, m1, m2 = vert_features
                lv.requires_grad = True
                optim = torch.optim.Adam([lv], lr=args.lr)

            losses.append(loss.cpu().item())
        t1 = time()

        if args.remesh_with_forward_at_end:
            with torch.no_grad():
                search_template, vert_features = remesh_template_from_deformed(
                    pred_mesh,
                    search_template,
                    vert_features=[best_lv],
                )
                best_lv = vert_features[0]
        t2 = time()

        with torch.no_grad():
            pred_mesh = decoder(search_template, best_lv, expand_lv=False)[-1]

            if args.remesh_at_end:
                pred_mesh = remesh_bk(pred_mesh)
        t3 = time()

        all_pred_meshes.append(pred_mesh.clone().cpu())
        
        # Store filename and optimized latent vector
        all_filenames.append(mesh_filenames[i])
        all_latent_vectors.append(best_lv.clone().cpu())

        true_point_samples = sample_points_from_meshes(true_mesh, 100000)
        pred_point_samples = sample_points_from_meshes(pred_mesh, 100000)

        metrics = point_metrics(true_point_samples, pred_point_samples, [0.01, 0.02])
        metrics[f'ChamferL2 x {cf}'] = chamfer_distance(true_point_samples, pred_point_samples)[0] * cf
        metrics['BL quality'] = 1.0 - mesh_bl_quality_loss(pred_mesh)
        if METRICS_AVAILABLE:
            metrics['No. ints.'] = 100 * float(self_intersections(pred_mesh.cpu())[0][0]) / len(pred_mesh.faces_packed())
        else:
            metrics['No. ints.'] = 0.0  # Fallback when C++ extension not available
        metrics['Search'] = t1 - t0
        metrics['Remesh'] = t2 - t1
        metrics['Decode'] = t3 - t2
        metrics['Total'] = t2 - t0

        all_metrics.append(metrics)

print('')
print('Done')

metrics = defaultdict(list)
for m in all_metrics:
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
        print('', file=stream)

    print('Validation', file=stream)
    print('-' * 50, file=stream)
    for k, v in metrics.items():
        vals = v[train_test_split_idx:]
        print(f'{k:20}: {vals.mean():8.4f} ± {vals.std():8.4f}', file=stream)
    print('', file=stream)

print_metrics_table(metrics, args.train_test_split_idx)
with open(join(args.output_dir, 'metrics.txt'), 'w') as f:
    print_metrics_table(metrics, args.train_test_split_idx, f)

with open(join(args.output_dir, 'args.txt'), 'w') as f:
    print(args, file=f)

# Save inference results with automatic suffix if file exists
base_out_fname = join(args.output_dir, 'inference_results.pt')
out_fname = base_out_fname
results_counter = 1
while os.path.exists(out_fname):
    out_fname = join(args.output_dir, f'inference_results_{results_counter}.pt')
    results_counter += 1

print('Writing results to:', out_fname)
torch.save({
    'pred_meshes': all_pred_meshes,
    'metrics': metrics,
    'args': args,
}, out_fname)


print(metrics)
# Save latent vectors and filenames with automatic suffix if file exists
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

# Print summary of saved files
print(f'\nFiles saved:')
print(f'  - Inference results: {os.path.basename(out_fname)}')
print(f'  - Latent vectors: {os.path.basename(latent_fname)}')
if results_counter > 1 or latent_counter > 1:
    print(f'  - Note: Files were saved with suffixes to avoid overwriting existing files')

print('All done. Exiting')
