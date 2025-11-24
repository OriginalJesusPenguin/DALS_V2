#!/usr/bin/env python3
"""
Single-model inference script for parallel processing.
Processes one checkpoint and writes results to individual CSV file.
"""

import torch
import os
import csv
import sys
import numpy as np
import argparse
from time import time
from glob import glob
from os.path import join, basename
from tqdm import tqdm
import trimesh

INFERENCE_RESULTS_DIR = "/home/ralbe/DALS/mesh_autodecoder/inference_results"

# Add project to path
sys.path.append('/home/ralbe/DALS/mesh_autodecoder')

# Pytorch3D imports
from pytorch3d.structures import Meshes
from pytorch3d.ops import (
    sample_points_from_meshes,
    sample_farthest_points,
    SubdivideMeshes,
)
from pytorch3d.loss import chamfer_distance
import pytorch3d.utils

# Project imports
from augment.point_wolf import augment_meshes
from util.data import load_meshes_in_dir
from model.mesh_decoder import MeshDecoder
from model.loss import mesh_bl_quality_loss, mesh_edge_loss_highdim, mesh_laplacian_loss_highdim, mesh_jacobian_determinant_loss

# Try importing metrics and remesh
try:
    from util.metrics import point_metrics, self_intersections
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import metrics module: {e}")
    METRICS_AVAILABLE = False
    def point_metrics(*_, **__): return {}
    def self_intersections(meshes): return torch.zeros(len(meshes)), torch.zeros(1)
try:
    from util.remesh import remesh_template_from_deformed, remesh_bk
except ImportError:
    def remesh_template_from_deformed(*_, **__):
        raise NotImplementedError("Remesh functionality not available - missing libremesh.so")
    def remesh_bk(*_, **__):
        raise NotImplementedError("Remesh functionality not available - missing libremesh.so")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optimize GPU memory usage
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # Enable mixed precision for faster inference
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Clear GPU cache
    torch.cuda.empty_cache()

# Args dataclass for cleaner config
class Args: pass

# Define inference configuration
def get_inference_args():
    args = Args()
    args.latent_mode = "global" 
    args.lr = 1e-3  # Increased learning rate for better optimization
    args.num_point_samples = 25_000  # Increased to 20k
    args.point_sample_mode = "fps"
    args.max_iters = 2000  # Fixed iterations with learning rate reduction
    args.template_subdiv = 4  
    args.remesh_with_forward_at_end = False
    args.remesh_at_end = False
    args.remesh_at = []
    args.train_test_split_idx = 0
    args.weight_bl_quality_loss = 1e-3
    args.weight_edge_length_loss = 1e-2  # Edge length regularization
    args.weight_laplacian_loss = 2e-3    # Laplacian regularization 
    args.weight_jacobian_det_loss = 5e-4 # Jacobian determinant loss to prevent self-intersections
    args.batch_size = 16  # Process multiple meshes in parallel
    args.init_strategy = "mean"  # Options: "random", "mean", "zero"
    return args

def save_mesh_to_obj(mesh, filepath):
    """Save a PyTorch3D mesh to OBJ file"""
    try:
        # Extract vertices and faces
        vertices = mesh.verts_packed().cpu().numpy()
        faces = mesh.faces_packed().cpu().numpy()
        
        # Create trimesh object
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Save as OBJ
        trimesh_mesh.export(filepath)
        return True
    except Exception as e:
        print(f"Error saving mesh to {filepath}: {e}")
        return False

# Function to run inference on a single checkpoint
def run_inference_on_checkpoint(checkpoint_info, args):
    """Run inference on a single checkpoint and return aggregated metrics"""
    
    # Load checkpoint
    checkpoint_path = checkpoint_info['DALS_path']
    print(f"Loading checkpoint: {os.path.basename(checkpoint_path)}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine test data path
    train_path = checkpoint_info['train_data_path']
    test_data_path = train_path.replace('train_meshes', 'test_meshes')
    mesh_output_dir = checkpoint_info["mesh_output_dir"]
    
    # Check if test data exists
    if not os.path.exists(test_data_path):
        print(f"Warning: Test data path does not exist: {test_data_path}")
        return None, None, mesh_output_dir
    
    # Load model
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
    decoder.to(device)
    
    latent_vectors = checkpoint['latent_vectors']
    latent_vectors.eval()
    latent_vectors.to(device)
    template = checkpoint['template']
    
    # Load test meshes
    print(f'Loading test data from: {test_data_path}')
    meshes = load_meshes_in_dir(test_data_path)
    print(f'Found {len(meshes)} test meshes')
    
    # Process all test samples
    print(f'Processing all {len(meshes)} test samples')
    
    # Create output directory for saved meshes
    mesh_output_dir = checkpoint_info["mesh_output_dir"]
    os.makedirs(mesh_output_dir, exist_ok=True)
    latent_output_dir = os.path.join(mesh_output_dir, 'latents')
    os.makedirs(latent_output_dir, exist_ok=True)
    print(f'Created mesh output directory: {mesh_output_dir}')
    
    if len(meshes) == 0:
        print("Warning: No test meshes found")
        return None, None, mesh_output_dir
    
    mesh_filenames = sorted(glob(os.path.join(test_data_path, '*.obj')))
    mesh_filenames = [basename(fname) for fname in mesh_filenames]
    
    # Determine remesh schedule
    remesh_at = []
    for r in args.remesh_at:
        remesh_at.append(r * args.max_iters if r < 1.0 else r)
    
    cf = 10000  # scale for reporting Chamfer
    
    # Run inference
    all_metrics = []
    print('Running inference...')
    
    if args.latent_mode == 'global':
        for i, (true_mesh, mesh_name) in enumerate(
            tqdm(zip(meshes, mesh_filenames), total=len(mesh_filenames), desc="Inference", ncols=100)
        ):
            true_mesh = true_mesh.to(device)
            true_points = sample_points_from_meshes(true_mesh, args.num_point_samples)
            
            # Initialize latent vector based on strategy
            if args.init_strategy == "random":
                lv = torch.randn(1, latent_vectors.weight.shape[1], device=device) * 0.1
            elif args.init_strategy == "mean":
                lv = latent_vectors.weight.data.mean(dim=0).clone().to(device).unsqueeze(0)
            elif args.init_strategy == "zero":
                lv = torch.zeros(1, latent_vectors.weight.shape[1], device=device)
            else:
                lv = torch.randn(1, latent_vectors.weight.shape[1], device=device) * 0.1
            lv.requires_grad_(True)
            optim = torch.optim.Adam([lv], lr=args.lr)
            search_template = pytorch3d.utils.ico_sphere(args.template_subdiv, device=device)
            search_template.scale_verts_(0.1)
            
            decoder.eval()
            decoder.requires_grad_(False)
            
            min_loss, no_improvement_iters = np.inf, 0
            best_iter, best_lv = -1, lv.clone()
            
            t0 = time()
            current_lr = args.lr
            lr_reduction_count = 0
            for it in range(args.max_iters):
                optim.zero_grad()
                pred_mesh = decoder(search_template, lv)[-1]
                pred_points = sample_points_from_meshes(pred_mesh, args.num_point_samples)
                loss = chamfer_distance(pred_points, true_points)[0]
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(lv, max_norm=1.0)
                
                if loss < min_loss:
                    min_loss = loss
                    no_improvement_iters, best_iter = 0, it
                    best_lv = lv.clone()  # Make a copy
                else:
                    no_improvement_iters += 1
                
                # Reduce learning rate if no improvement for 100 iterations
                if no_improvement_iters > 100 and lr_reduction_count < 3:  # Max 3 reductions
                    current_lr *= 0.5  # Reduce by half
                    for param_group in optim.param_groups:
                        param_group['lr'] = current_lr
                    no_improvement_iters = 0  # Reset counter
                    lr_reduction_count += 1
                    print(f"  Reduced learning rate to {current_lr:.2e} at iter {it}")
                
                optim.step()
                
                # Print progress every 50 iterations
                if it % 250 == 0:
                    print(f"  Iter {it}: Loss = {loss.item():.6f}, Best = {min_loss:.6f}, LR = {current_lr:.2e}")
                
                if it in remesh_at:
                    with torch.no_grad():
                        search_template = remesh_template_from_deformed(pred_mesh, search_template)
            t1 = time()
            print(f"  Final loss: {min_loss:.6f} (best at iter {best_iter})")
            
            if args.remesh_with_forward_at_end:
                with torch.no_grad():
                    search_template = remesh_template_from_deformed(pred_mesh, search_template, ratio=0.6)
            t2 = time()
            
            with torch.no_grad():
                pred_mesh = decoder(search_template, best_lv)[-1]
                if args.remesh_at_end:
                    v0, v1 = pred_mesh.verts_packed()[pred_mesh.edges_packed()].unbind(1)
                    h = torch.norm(v1 - v0, dim=1).mean().cpu()
                    pred_mesh = remesh_bk(pred_mesh, target_length=h, iters=5)
            t3 = time()
            
            # Compute metrics
            true_point_samples = sample_points_from_meshes(true_mesh, 10000)  # Keep at 10k for metrics
            pred_point_samples = sample_points_from_meshes(pred_mesh, 10000)  # Keep at 10k for metrics
            metrics = point_metrics(true_point_samples, pred_point_samples, [0.01, 0.02])
            metrics[f'ChamferL2 x {cf}'] = chamfer_distance(true_point_samples, pred_point_samples)[0] * cf
            metrics['BL quality'] = 1.0 - mesh_bl_quality_loss(pred_mesh)
            if METRICS_AVAILABLE:
                ints = 100 * float(self_intersections(pred_mesh.cpu())[0][0]) / len(pred_mesh.faces_packed())
                metrics['No. ints.'] = ints
            else:
                metrics['No. ints.'] = 0.0
            metrics['Search'] = t1 - t0
            metrics['Remesh'] = t2 - t1
            metrics['Decode'] = t3 - t2
            metrics['Total'] = t3 - t0
            metrics['target_mesh_name'] = mesh_name
            metrics_record = {}
            for key, value in metrics.items():
                if hasattr(value, 'item'):
                    value = value.item()
                if isinstance(value, np.generic):
                    value = float(value)
                if isinstance(value, (float, int)):
                    metrics_record[key] = float(value)
                else:
                    metrics_record[key] = value
            all_metrics.append(metrics_record)
            
            # Save the final optimized mesh
            mesh_stem = os.path.splitext(mesh_name)[0]
            optimized_filename = f'{mesh_stem}_optimized.obj'
            optimized_path = os.path.join(mesh_output_dir, optimized_filename)
            if save_mesh_to_obj(pred_mesh, optimized_path):
                print(f'Saved optimized mesh {i+1}/{len(meshes)}: {optimized_filename}')

            target_filename = f'{mesh_stem}_target.obj'
            target_path = os.path.join(mesh_output_dir, target_filename)
            if save_mesh_to_obj(true_mesh, target_path):
                print(f'Saved target mesh {i+1}/{len(meshes)}: {target_filename}')

            latent_filename = f'{mesh_stem}_latent.pt'
            latent_path = os.path.join(latent_output_dir, latent_filename)
            torch.save(best_lv.detach().cpu(), latent_path)
            print(f'Saved latent vector {i+1}/{len(meshes)}: {latent_filename}')
            
            # Clear GPU cache to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    elif args.latent_mode == 'local':
        for i, (true_mesh, mesh_name) in enumerate(
            tqdm(zip(meshes, mesh_filenames), total=len(mesh_filenames), desc="Local latent optimization")
        ):
            true_mesh = true_mesh.to(device)
            if args.point_sample_mode == 'uniform':
                true_points = sample_points_from_meshes(true_mesh, args.num_point_samples)
            else:
                num_init_samples = max(10000, args.num_point_samples * 100)
                all_true_points = sample_points_from_meshes(true_mesh, num_init_samples)
                true_points = sample_farthest_points(all_true_points, K=args.num_point_samples)[0]
            
            search_template = pytorch3d.utils.ico_sphere(args.template_subdiv, device=device)
            search_template.scale_verts_(0.1)
            
            lv = latent_vectors.weight.data.mean(dim=0).clone().to(device).unsqueeze(0)
            lv = lv.repeat(len(search_template.verts_packed()), 1)
            lv.requires_grad_(True)
            
            decoder.eval()
            decoder.requires_grad_(False)
            
            optim = torch.optim.Adam([lv], lr=args.lr)
            min_loss, no_improvement_iters = np.inf, 0
            best_iter, best_lv = -1, lv.clone()
            
            m1, m2 = torch.zeros_like(lv), torch.zeros_like(lv)
            
            t0 = time()
            current_lr = args.lr
            lr_reduction_count = 0
            for it in range(args.max_iters):
                optim.zero_grad()
                lv.requires_grad_(True)
                pred_mesh = decoder(search_template, lv, expand_lv=False)[-1]
                pred_points = sample_points_from_meshes(pred_mesh, args.num_point_samples)
                
                # Core loss
                loss = chamfer_distance(pred_points, true_points)[0]
                loss += args.weight_bl_quality_loss * mesh_bl_quality_loss(pred_mesh)
                loss += args.weight_edge_length_loss * mesh_edge_loss_highdim(pred_mesh, lv)
                loss += args.weight_laplacian_loss * mesh_laplacian_loss_highdim(pred_mesh, lv)
                # Note: We use search_template (current template) for Jacobian loss,
                # not original_template, because remeshing changes the mesh topology
                # The Jacobian measures local deformation from current template to current prediction
                loss += args.weight_jacobian_det_loss * mesh_jacobian_determinant_loss(search_template, pred_mesh)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(lv, max_norm=1.0)
                
                if loss < min_loss:
                    min_loss = loss
                    no_improvement_iters, best_iter = 0, it
                    best_lv = lv.clone()  # Make a copy
                else:
                    no_improvement_iters += 1
                
                # Reduce learning rate if no improvement for 100 iterations
                if no_improvement_iters > 100 and lr_reduction_count < 3:  # Max 3 reductions
                    current_lr *= 0.5  # Reduce by half
                    for param_group in optim.param_groups:
                        param_group['lr'] = current_lr
                    no_improvement_iters = 0  # Reset counter
                    lr_reduction_count += 1
                    print(f"  Reduced learning rate to {current_lr:.2e} at iter {it}")
                
                optim.step()
                
                # Print progress every 50 iterations
                if it % 250 == 0:
                    print(f"  Iter {it}: Loss = {loss.item():.6f}, Best = {min_loss:.6f}, LR = {current_lr:.2e}")
                
                if it in remesh_at:
                    with torch.no_grad():
                        search_template, vert_features = remesh_template_from_deformed(
                            pred_mesh, search_template, vert_features=[lv, best_lv, m1, m2])
                        lv, best_lv, m1, m2 = vert_features
                    lv.requires_grad = True
                    optim = torch.optim.Adam([lv], lr=current_lr)  # Use current learning rate
            t1 = time()
            print(f"  Final loss: {min_loss:.6f} (best at iter {best_iter})")
            
            if args.remesh_with_forward_at_end:
                with torch.no_grad():
                    search_template, vert_features = remesh_template_from_deformed(
                        pred_mesh, search_template, vert_features=[best_lv])
                    best_lv = vert_features[0]
            t2 = time()
            
            with torch.no_grad():
                pred_mesh = decoder(search_template, best_lv, expand_lv=False)[-1]
                if args.remesh_at_end:
                    pred_mesh = remesh_bk(pred_mesh)
            t3 = time()
            
            # Compute metrics
            true_point_samples = sample_points_from_meshes(true_mesh, 2500)  # Keep at 10k for metrics
            pred_point_samples = sample_points_from_meshes(pred_mesh, 2500)  # Keep at 10k for metrics
            metrics = point_metrics(true_point_samples, pred_point_samples, [0.01, 0.02])
            metrics[f'ChamferL2 x {cf}'] = chamfer_distance(true_point_samples, pred_point_samples)[0] * cf
            metrics['BL quality'] = 1.0 - mesh_bl_quality_loss(pred_mesh)
            if METRICS_AVAILABLE:
                ints = 100 * float(self_intersections(pred_mesh.cpu())[0][0]) / len(pred_mesh.faces_packed())
                metrics['No. ints.'] = ints
            else:
                metrics['No. ints.'] = 0.0
            metrics['Search'] = t1 - t0
            metrics['Remesh'] = t2 - t1
            metrics['Decode'] = t3 - t2
            metrics['Total'] = t2 - t0
            metrics['target_mesh_name'] = mesh_name
            metrics_record = {}
            for key, value in metrics.items():
                if hasattr(value, 'item'):
                    value = value.item()
                if isinstance(value, np.generic):
                    value = float(value)
                if isinstance(value, (float, int)):
                    metrics_record[key] = float(value)
                else:
                    metrics_record[key] = value
            all_metrics.append(metrics_record)
            
            # Save the final optimized mesh
            mesh_stem = os.path.splitext(mesh_name)[0]
            optimized_filename = f'{mesh_stem}_optimized.obj'
            optimized_path = os.path.join(mesh_output_dir, optimized_filename)
            if save_mesh_to_obj(pred_mesh, optimized_path):
                print(f'Saved optimized mesh {i+1}/{len(meshes)}: {optimized_filename}')

            target_filename = f'{mesh_stem}_target.obj'
            target_path = os.path.join(mesh_output_dir, target_filename)
            if save_mesh_to_obj(true_mesh, target_path):
                print(f'Saved target mesh {i+1}/{len(meshes)}: {target_filename}')

            latent_filename = f'{mesh_stem}_latent.pt'
            latent_path = os.path.join(latent_output_dir, latent_filename)
            torch.save(best_lv.detach().cpu(), latent_path)
            print(f'Saved latent vector {i+1}/{len(meshes)}: {latent_filename}')
            
            # Clear GPU cache to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Aggregate metrics
    if len(all_metrics) == 0:
        return None, None, mesh_output_dir
    
    # Compute mean and std for each metric
    aggregated_metrics = {}
    exclude_from_aggregation = {"target_mesh_name", "Search", "Total"}
    aggregation_keys = [
        key
        for key, value in all_metrics[0].items()
        if key not in exclude_from_aggregation and isinstance(value, (int, float))
    ]

    for metric_name in aggregation_keys:
        values = [m[metric_name] for m in all_metrics if isinstance(m.get(metric_name), (int, float))]
        if not values:
            continue
        aggregated_metrics[f'{metric_name}_mean'] = float(np.mean(values))
        aggregated_metrics[f'{metric_name}_std'] = float(np.std(values))

    aggregated_metrics['num_test_samples'] = len(all_metrics)
    
    # Print summary of saved meshes
    saved_meshes = [f for f in os.listdir(mesh_output_dir) if f.endswith('.obj')]
    optimized_count = len([f for f in saved_meshes if f.endswith('_optimized.obj')])
    target_count = len([f for f in saved_meshes if f.endswith('_target.obj')])
    print(f'Saved {optimized_count} optimized meshes and {target_count} target meshes to {mesh_output_dir}')
    
    return aggregated_metrics, all_metrics, mesh_output_dir

def main():
    parser = argparse.ArgumentParser(description='Run inference on a single model checkpoint')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the checkpoint file')

    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Available GPU Memory: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
    else:
        print("Warning: CUDA not available, using CPU (will be very slow)")

    print(f"Processing checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    latent_vectors = checkpoint['latent_vectors']
    train_data_path = checkpoint['train_data_path']
    decoder_mode = checkpoint['decoder_mode']

    if isinstance(latent_vectors, torch.nn.Embedding):
        latent_dim = latent_vectors.weight.shape[1]
    else:
        latent_dim = None

    model_name = os.path.basename(args.checkpoint_path).replace('.ckpt', '')
    # model_name = 'MeshDecoderTrainer_2025-11-21_11-44-53.ckpt'
    mesh_output_dir = os.path.join(INFERENCE_RESULTS_DIR, f'meshes_{model_name}')

    checkpoint_info = {
        "DALS_path": args.checkpoint_path,
        "decoder_mode": decoder_mode,
        "latent_dim": latent_dim,
        "train_data_path": train_data_path,
        "model_name": model_name,
        "mesh_output_dir": mesh_output_dir,
    }

    inference_args = get_inference_args()
    aggregated_metrics, per_sample_metrics, mesh_output_dir = run_inference_on_checkpoint(checkpoint_info, inference_args)

    os.makedirs(mesh_output_dir, exist_ok=True)

    aggregated_row = {
        "checkpoint_path": args.checkpoint_path,
        "decoder_mode": decoder_mode,
        "latent_dim": latent_dim if latent_dim is not None else '',
        "latent_mode": inference_args.latent_mode,
        "lr": inference_args.lr,
        "max_iters": inference_args.max_iters,
        "template_subdiv": inference_args.template_subdiv,
        "weight_bl_quality_loss": inference_args.weight_bl_quality_loss,
        "weight_edge_length_loss": inference_args.weight_edge_length_loss,
        "weight_laplacian_loss": inference_args.weight_laplacian_loss,
        "weight_jacobian_det_loss": inference_args.weight_jacobian_det_loss,
    }

    aggregated_path = os.path.join(mesh_output_dir, "test_metrics.csv")
    per_sample_path = os.path.join(mesh_output_dir, "test_metrics_per_sample.csv")

    if aggregated_metrics is not None:
        print(f"Successfully processed {aggregated_metrics['num_test_samples']} test samples")

        metric_keys = sorted(key for key in aggregated_metrics.keys() if key != 'num_test_samples')
        for key in metric_keys:
            aggregated_row[key] = aggregated_metrics[key]
        aggregated_row['num_test_samples'] = aggregated_metrics.get('num_test_samples', 0)

        aggregated_columns = list(aggregated_row.keys())
        with open(aggregated_path, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=aggregated_columns)
            writer.writeheader()
            writer.writerow(aggregated_row)

        if per_sample_metrics:
            metric_columns = sorted({
                key for row in per_sample_metrics for key in row.keys() if key != 'target_mesh_name'
            })
            per_sample_columns = ['target_mesh_name'] + metric_columns
            with open(per_sample_path, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=per_sample_columns)
                writer.writeheader()
                for row in per_sample_metrics:
                    cleaned_row = {}
                    for field in per_sample_columns:
                        value = row.get(field, '')
                        cleaned_row[field] = value
                    writer.writerow(cleaned_row)

        if 'ChamferL2 x 10000_mean' in aggregated_metrics:
            print(f"ChamferL2: {aggregated_metrics['ChamferL2 x 10000_mean']:.4f} ± {aggregated_metrics['ChamferL2 x 10000_std']:.4f}")
        if 'BL quality_mean' in aggregated_metrics:
            print(f"BL Quality: {aggregated_metrics['BL quality_mean']:.4f} ± {aggregated_metrics['BL quality_std']:.4f}")
    else:
        print("Failed to process this checkpoint")
        aggregated_row['num_test_samples'] = 0
        aggregated_columns = list(aggregated_row.keys())
        with open(aggregated_path, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=aggregated_columns)
            writer.writeheader()
            writer.writerow(aggregated_row)

        with open(per_sample_path, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['target_mesh_name'])
            writer.writeheader()

    print(f"Aggregated metrics saved to {aggregated_path}")
    print(f"Per-sample metrics saved to {per_sample_path}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Final GPU Memory Usage: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")

if __name__ == "__main__":
    main()
