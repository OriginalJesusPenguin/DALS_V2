import torch
import glob
import os
import csv
import sys
import numpy as np
from collections import defaultdict
from time import time
from glob import glob
from os.path import join, basename
from tqdm import tqdm

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
from model.loss import mesh_bl_quality_loss, mesh_edge_loss_highdim, mesh_laplacian_loss_highdim

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

# Load all checkpoints and extract parameters
models_folder = '/home/ralbe/DALS/mesh_autodecoder/models'
all_ckpt_files = glob(os.path.join(models_folder, '*.ckpt'))
models_list = [f for f in all_ckpt_files if 'MeshDecoderTrainer' in os.path.basename(f)]
models_list = models_list

print(f"Found {len(models_list)} checkpoints to process")

results = []
for model_file in models_list:
    loaded_checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    latent_vectors = loaded_checkpoint['latent_vectors']
    train_data_path = loaded_checkpoint['train_data_path']
    decoder_mode = loaded_checkpoint['decoder_mode']

    # Extract the parent folder name one above 'train_meshes'
    folder_name = os.path.basename(os.path.dirname(train_data_path))
    if 'mixed' in folder_name:
        split_type = 'mixed'
    elif 'separate' in folder_name:
        split_type = 'separate'
    else:
        split_type = None

    if 'global' in folder_name:
        scaling_type = 'global'
    elif 'individual' in folder_name:
        scaling_type = 'individual'
    else:
        scaling_type = None

    if 'noaug' in folder_name:
        augmentation_type = 'noaug'
    elif 'aug' in folder_name:
        augmentation_type = 'aug'
    else:
        augmentation_type = None

    if isinstance(latent_vectors, torch.nn.Embedding):
        latent_dim = latent_vectors.weight.shape[1]
    else:
        latent_dim = None

    print(f"split_type: {split_type}, scaling_type: {scaling_type}, augmentation_type: {augmentation_type}, decoder_mode: {decoder_mode}, latent_dim: {latent_dim}")

    results.append({
        "DALS_path": model_file,
        "split_type": split_type,
        "augmentation_type": augmentation_type,
        "scaling_type": scaling_type,
        "decoder_mode": decoder_mode,
        "latent_dim": latent_dim,
        "train_data_path": train_data_path
    })




# Args dataclass for cleaner config
class Args: pass

# Define inference configuration
def get_inference_args():
    args = Args()
    args.latent_mode = "local" 
    args.lr = 1e-3
    args.num_point_samples = 2500
    args.point_sample_mode = "fps"
    args.max_iters = 250
    args.template_subdiv = 4
    args.remesh_with_forward_at_end = False
    args.remesh_at_end = False
    args.remesh_at = []
    args.train_test_split_idx = 0
    args.weight_bl_quality_loss = 1e-3
    args.weight_edge_length_loss = 1e2
    args.batch_size = 64  # Process multiple meshes in parallel
    return args

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
    
    # Check if test data exists
    if not os.path.exists(test_data_path):
        print(f"Warning: Test data path does not exist: {test_data_path}")
        return None
    
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
    template = checkpoint['template']
    
    # Load test meshes
    print(f'Loading test data from: {test_data_path}')
    meshes = load_meshes_in_dir(test_data_path)
    print(f'Found {len(meshes)} test meshes')
    
    if len(meshes) == 0:
        print("Warning: No test meshes found")
        return None
    
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
        for i, true_mesh in enumerate(tqdm(meshes, desc="Inference", ncols=100)):
            true_mesh = true_mesh.to(device)
            true_points = sample_points_from_meshes(true_mesh, args.num_point_samples)
            
            lv = latent_vectors.weight.data.mean(dim=0).clone().to(device).unsqueeze(0)
            lv.requires_grad_(True)
            optim = torch.optim.Adam([lv], lr=args.lr)
            search_template = pytorch3d.utils.ico_sphere(args.template_subdiv, device=device)
            search_template.scale_verts_(0.1)
            
            decoder.eval()
            decoder.requires_grad_(False)
            
            min_loss, no_improvement_iters = np.inf, 0
            best_iter, best_lv = -1, lv
            
            t0 = time()
            for it in range(args.max_iters):
                optim.zero_grad()
                pred_mesh = decoder(search_template, lv)[-1]
                pred_points = sample_points_from_meshes(pred_mesh, args.num_point_samples)
                loss = chamfer_distance(pred_points, true_points)[0]
                loss.backward()
                
                if loss < min_loss:
                    min_loss = loss
                    no_improvement_iters, best_iter = 0, it
                    best_lv = lv
                else:
                    no_improvement_iters += 1
                if no_improvement_iters > 10: break
                optim.step()
                
                if it in remesh_at:
                    with torch.no_grad():
                        search_template = remesh_template_from_deformed(pred_mesh, search_template)
            t1 = time()
            
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
            true_point_samples = sample_points_from_meshes(true_mesh, 100000)
            pred_point_samples = sample_points_from_meshes(pred_mesh, 100000)
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
            all_metrics.append(metrics)
    
    elif args.latent_mode == 'local':
        for i, true_mesh in enumerate(tqdm(meshes, desc="Local latent optimization")):
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
            best_iter, best_lv = -1, lv
            
            m1, m2 = torch.zeros_like(lv), torch.zeros_like(lv)
            
            t0 = time()
            for it in range(args.max_iters):
                optim.zero_grad()
                lv.requires_grad_(True)
                pred_mesh = decoder(search_template, lv, expand_lv=False)[-1]
                pred_points = sample_points_from_meshes(pred_mesh, args.num_point_samples)
                
                # Core loss
                loss = chamfer_distance(pred_points, true_points)[0]
                loss += args.weight_bl_quality_loss * mesh_bl_quality_loss(pred_mesh)
                loss += args.weight_edge_length_loss * mesh_laplacian_loss_highdim(pred_mesh, lv)
                loss.backward()
                
                if loss < 1.05 * min_loss:
                    min_loss = loss
                    no_improvement_iters, best_iter = 0, it
                    best_lv = lv
                else:
                    no_improvement_iters += 1
                if no_improvement_iters > 10:
                    pass
                
                optim.step()
                
                if it in remesh_at:
                    with torch.no_grad():
                        search_template, vert_features = remesh_template_from_deformed(
                            pred_mesh, search_template, vert_features=[lv, best_lv, m1, m2])
                        lv, best_lv, m1, m2 = vert_features
                    lv.requires_grad = True
                    optim = torch.optim.Adam([lv], lr=args.lr)
            t1 = time()
            
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
            true_point_samples = sample_points_from_meshes(true_mesh, 100000)
            pred_point_samples = sample_points_from_meshes(pred_mesh, 100000)
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
            all_metrics.append(metrics)
    
    # Aggregate metrics
    if len(all_metrics) == 0:
        return None
    
    # Compute mean and std for each metric
    aggregated_metrics = {}
    for metric_name in all_metrics[0].keys():
        values = [m[metric_name] for m in all_metrics]
        # Convert PyTorch tensors to Python floats before computing statistics
        float_values = []
        for v in values:
            if hasattr(v, 'item'):  # PyTorch tensor
                float_values.append(v.item())
            else:  # Already a Python float/int
                float_values.append(float(v))
        aggregated_metrics[f'{metric_name}_mean'] = np.mean(float_values)
        aggregated_metrics[f'{metric_name}_std'] = np.std(float_values)
    
    aggregated_metrics['num_test_samples'] = len(all_metrics)
    
    return aggregated_metrics

# Main execution: iterate through all checkpoints
print(f"\n{'='*80}")
print(f"Starting inference sweep on {len(results)} checkpoints")
print(f"{'='*80}\n")

args = get_inference_args()

# Process each checkpoint
for idx, checkpoint_info in enumerate(results):
    print(f"\n{'='*80}")
    print(f"Processing checkpoint {idx+1}/{len(results)}")
    print(f"Model: {os.path.basename(checkpoint_info['DALS_path'])}")
    print(f"Config: {checkpoint_info['split_type']}_{checkpoint_info['scaling_type']}_{checkpoint_info['augmentation_type']}")
    print(f"Decoder: {checkpoint_info['decoder_mode']}, Latent: {checkpoint_info['latent_dim']}")
    print(f"{'='*80}\n")
    
    # Run inference and get metrics
    metrics = run_inference_on_checkpoint(checkpoint_info, args)
    
    if metrics is not None:
        # Update checkpoint info with metrics
        checkpoint_info.update(metrics)
        print(f"Successfully processed {checkpoint_info['num_test_samples']} test samples")
        
        # Print some key metrics
        if 'ChamferL2 x 10000_mean' in metrics:
            print(f"ChamferL2: {metrics['ChamferL2 x 10000_mean']:.4f} ± {metrics['ChamferL2 x 10000_std']:.4f}")
        if 'BL quality_mean' in metrics:
            print(f"BL Quality: {metrics['BL quality_mean']:.4f} ± {metrics['BL quality_std']:.4f}")
    else:
        print("Failed to process this checkpoint")
        # Add placeholder metrics
        checkpoint_info.update({
            'num_test_samples': 0,
            'ChamferL2 x 10000_mean': None,
            'ChamferL2 x 10000_std': None,
            'BL quality_mean': None,
            'BL quality_std': None,
        })

# Define extended CSV columns
csv_columns_extended = [
    "DALS_path", "split_type", "augmentation_type", "scaling_type", "decoder_mode", "latent_dim",
    "ChamferL2 x 10000_mean", "ChamferL2 x 10000_std",
    "BL quality_mean", "BL quality_std", 
    "No. ints._mean", "No. ints._std",
    "Precision@0.01_mean", "Precision@0.01_std",
    "Recall@0.01_mean", "Recall@0.01_std", 
    "F1@0.01_mean", "F1@0.01_std",
    "Precision@0.02_mean", "Precision@0.02_std",
    "Recall@0.02_mean", "Recall@0.02_std",
    "F1@0.02_mean", "F1@0.02_std",
    "Search_mean", "Search_std",
    "Total_mean", "Total_std",
    "num_test_samples"
]

# Write final CSV with all results
csv_filename = "model_inference_summary.csv"
print(f"\n{'='*80}")
print(f"Saving results to {csv_filename}")
print(f"{'='*80}\n")

with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns_extended)
    writer.writeheader()
    for row in results:
        # Clean the row data to handle None values and ensure all fields are present
        cleaned_row = {}
        for field in csv_columns_extended:
            if field in row and row[field] is not None:
                cleaned_row[field] = row[field]
            else:
                cleaned_row[field] = ''  # Use empty string for missing/None values
        writer.writerow(cleaned_row)

print(f"Successfully saved inference results for {len(results)} checkpoints to {csv_filename}")
print("All done. Exiting")
