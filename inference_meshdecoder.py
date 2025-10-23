import os
import sys
import argparse
from collections import defaultdict
from time import time
from os.path import join

import numpy as np
import torch
from tqdm import tqdm
import trimesh

import pytorch3d
import pytorch3d.utils
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
from pytorch3d.loss import chamfer_distance

from util.data import load_meshes_in_dir
from model.mesh_decoder import MeshDecoder

# Import metrics with error handling
try:
    from util.metrics import point_metrics, self_intersections
    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import metrics module: {e}")
    METRICS_AVAILABLE = False
    
    def point_metrics(*args, **kwargs):
        return {}
    
    def self_intersections(meshes):
        return torch.zeros(len(meshes)), torch.zeros(1)

try:
    from util.remesh import remesh_template_from_deformed, remesh_bk
except ImportError:
    def remesh_template_from_deformed(*args, **kwargs):
        raise NotImplementedError("Remesh functionality not available - missing libremesh.so")
    def remesh_bk(*args, **kwargs):
        raise NotImplementedError("Remesh functionality not available - missing libremesh.so")

from model.loss import mesh_bl_quality_loss, mesh_edge_loss_highdim, mesh_laplacian_loss_highdim

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Mesh decoder inference script')
    parser.add_argument('--data_path', required=True, help='Path to mesh data directory')
    parser.add_argument('--checkpoint_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default='.', help='Output directory for results')
    parser.add_argument('--latent_mode', choices=['global', 'local'], required=True,
                       help='Latent optimization mode')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_point_samples', type=int, default=5000,
                       help='Number of point samples for evaluation')
    parser.add_argument('--point_sample_mode', choices=['uniform', 'fps'],
                       default='fps', help='Point sampling strategy')
    parser.add_argument('--max_iters', type=int, default=250,
                       help='Maximum optimization iterations')
    parser.add_argument('--template_subdiv', type=int, default=4,
                       help='Template subdivision level')
    parser.add_argument('--remesh_with_forward_at_end', action='store_true',
                       help='Remesh with forward pass at end')
    parser.add_argument('--remesh_at_end', action='store_true',
                       help='Remesh at the end')
    parser.add_argument('--remesh_at', type=float, default=[], nargs='*',
                       help='Iterations at which to remesh')
    parser.add_argument('--train_test_split_idx', type=int, default=0,
                       help='Index for train/test split')
    parser.add_argument('--weight_bl_quality_loss', type=float, default=1e-3,
                       help='Weight for BL quality loss')
    parser.add_argument('--weight_edge_length_loss', type=float, default=1e2,
                       help='Weight for edge length loss')
    return parser.parse_args()


def load_model_and_data(args):
    """Load model checkpoint and mesh data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    print(f'Loading checkpoint: {args.checkpoint_path}')
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    hparams = checkpoint['hparams']
    
    # Initialize decoder
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
    
    # Load latent vectors and template
    latent_vectors = checkpoint['latent_vectors']
    latent_vectors.eval()
    template = checkpoint['template']
    
    # Load mesh data
    print(f'Loading data from: {args.data_path}')
    meshes = load_meshes_in_dir(args.data_path)
    print(f'Found {len(meshes)} meshes')
    
    # Get mesh filenames
    import glob
    mesh_filenames = sorted(glob.glob(os.path.join(args.data_path, '*.obj')))
    mesh_filenames = [os.path.basename(fname) for fname in mesh_filenames]
    print(f'Found {len(mesh_filenames)} filenames')
    
    return device, decoder, latent_vectors, template, meshes, mesh_filenames

def process_remesh_iterations(args):
    """Process remesh iteration parameters."""
    remesh_at = []
    for r in args.remesh_at:
        if r < 1.0:
            remesh_at.append(r * args.max_iters)
        else:
            remesh_at.append(r)
    return remesh_at


def save_mesh_to_obj(mesh, filepath):
    """Save PyTorch3D mesh to OBJ file using trimesh."""
    try:
        # Convert PyTorch3D mesh to trimesh
        vertices = mesh.verts_packed().cpu().numpy()
        faces = mesh.faces_packed().cpu().numpy()
        
        # Create trimesh object
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Save as OBJ
        trimesh_mesh.export(filepath)
        return True
    except Exception as e:
        print(f"Warning: Could not save mesh to {filepath}: {e}")
        return False


def run_global_inference(device, decoder, latent_vectors, template, meshes, mesh_filenames, args, remesh_at):
    """Run global latent optimization inference."""
    all_metrics = []
    all_pred_meshes = []
    all_latent_vectors = []
    all_filenames = []
    
    # Create mesh output directory
    mesh_output_dir = join(args.output_dir, 'meshes')
    os.makedirs(mesh_output_dir, exist_ok=True)
    
    print('Running global inference')
    
    for i, true_mesh in enumerate(tqdm(meshes, desc="Global Inference", ncols=100)):
        true_mesh = true_mesh.to(device)
        true_points = sample_points_from_meshes(true_mesh, args.num_point_samples)

        # Initialize latent vector
        lv = latent_vectors.weight.data.mean(dim=0).clone().to(device).unsqueeze(0)
        lv.requires_grad_(True)
        optim = torch.optim.Adam([lv], lr=args.lr)

        # Create search template
        search_template = pytorch3d.utils.ico_sphere(args.template_subdiv, device=device)
        search_template.scale_verts_(0.1)

        decoder.eval()
        decoder.requires_grad_(False)
        decoder.to(device)

        # Optimization loop
        min_loss = np.inf
        no_improvement_iters = 0
        best_lv = lv

        t0 = time()
        for it in range(args.max_iters):
            optim.zero_grad()
            pred_mesh = decoder(search_template, lv)[-1]
            pred_points = sample_points_from_meshes(pred_mesh, args.num_point_samples)
            loss = chamfer_distance(pred_points, true_points)[0]
            loss.backward()

            if loss < min_loss:
                min_loss = loss
                no_improvement_iters = 0
                best_lv = lv
            else:
                no_improvement_iters += 1

            if no_improvement_iters > 10:
                break

            optim.step()

            # Remesh if needed
            if it in remesh_at:
                with torch.no_grad():
                    search_template = remesh_template_from_deformed(pred_mesh, search_template)

        t1 = time()

        # Final remesh if requested
        if args.remesh_with_forward_at_end:
            with torch.no_grad():
                search_template = remesh_template_from_deformed(
                    pred_mesh, search_template, ratio=0.6
                )
        t2 = time()

        # Generate final mesh
        with torch.no_grad():
            pred_mesh = decoder(search_template, best_lv)[-1]
            if args.remesh_at_end:
                v0, v1 = pred_mesh.verts_packed()[pred_mesh.edges_packed()].unbind(1)
                h = torch.norm(v1 - v0, dim=1).mean().cpu()
                pred_mesh = remesh_bk(pred_mesh, target_length=h, iters=5)
        t3 = time()

        # Store results
        all_pred_meshes.append(pred_mesh.clone().cpu())
        all_filenames.append(mesh_filenames[i])
        all_latent_vectors.append(best_lv.clone().cpu())

        # Save optimized mesh
        mesh_filename = mesh_filenames[i]
        mesh_name = os.path.splitext(mesh_filename)[0]  # Remove .obj extension
        output_mesh_path = join(mesh_output_dir, f'{mesh_name}_optimized.obj')
        save_mesh_to_obj(pred_mesh, output_mesh_path)

        # Compute metrics
        true_point_samples = sample_points_from_meshes(true_mesh, 100000)
        pred_point_samples = sample_points_from_meshes(pred_mesh, 100000)

        metrics = point_metrics(true_point_samples, pred_point_samples, [0.01, 0.02])
        metrics['ChamferL2 x 10000'] = chamfer_distance(true_point_samples, pred_point_samples)[0] * 10000
        metrics['BL quality'] = 1.0 - mesh_bl_quality_loss(pred_mesh)
        
        if METRICS_AVAILABLE:
            metrics['No. ints.'] = 100 * float(self_intersections(pred_mesh.cpu())[0][0]) / len(pred_mesh.faces_packed())
        else:
            metrics['No. ints.'] = 0.0
            
        metrics['Search'] = t1 - t0
        metrics['Remesh'] = t2 - t1
        metrics['Decode'] = t3 - t2
        metrics['Total'] = t3 - t0

        all_metrics.append(metrics)
    
    return all_metrics, all_pred_meshes, all_latent_vectors, all_filenames


def run_local_inference(device, decoder, latent_vectors, template, meshes, mesh_filenames, args, remesh_at):
    """Run local latent optimization inference."""
    all_metrics = []
    all_pred_meshes = []
    all_latent_vectors = []
    all_filenames = []
    
    # Create mesh output directory
    mesh_output_dir = join(args.output_dir, 'meshes')
    os.makedirs(mesh_output_dir, exist_ok=True)
    
    print('Running local inference')
    
    for i, true_mesh in enumerate(tqdm(meshes, desc="Local Inference", ncols=100)):
        true_mesh = true_mesh.to(device)
        
        # Sample points
        if args.point_sample_mode == 'uniform':
            true_points = sample_points_from_meshes(true_mesh, args.num_point_samples)
        else:  # fps
            num_init_samples = max(10000, args.num_point_samples * 100)
            all_true_points = sample_points_from_meshes(true_mesh, num_init_samples)
            true_points = sample_farthest_points(all_true_points, K=args.num_point_samples)[0]

        # Create search template
        search_template = pytorch3d.utils.ico_sphere(args.template_subdiv, device=device)
        search_template.scale_verts_(0.1)

        # Initialize latent vectors
        lv = latent_vectors.weight.data.mean(dim=0).clone().to(device).unsqueeze(0)
        lv = lv.repeat(len(search_template.verts_packed()), 1)
        lv.requires_grad_(True)

        decoder.eval()
        decoder.requires_grad_(False)
        decoder.to(device)
        optim = torch.optim.Adam([lv], lr=args.lr)

        # Optimization loop
        min_loss = np.inf
        no_improvement_iters = 0
        best_lv = lv

        t0 = time()
        for it in range(args.max_iters):
            optim.zero_grad()
            lv.requires_grad_(True)
            pred_mesh = decoder(search_template, lv, expand_lv=False)[-1]

            pred_points = sample_points_from_meshes(pred_mesh, args.num_point_samples)
            loss = chamfer_distance(pred_points, true_points)[0]
            loss += args.weight_bl_quality_loss * mesh_bl_quality_loss(pred_mesh)
            loss += args.weight_edge_length_loss * mesh_laplacian_loss_highdim(pred_mesh, lv)
            
            loss.backward()

            if loss < 1.05 * min_loss:
                min_loss = loss
                no_improvement_iters = 0
                best_lv = lv
            else:
                no_improvement_iters += 1

            if no_improvement_iters > 20:
                pass  # Continue optimization

            optim.step()

            # Remesh if needed
            if it in remesh_at:
                with torch.no_grad():
                    search_template, vert_features = remesh_template_from_deformed(
                        pred_mesh, search_template, vert_features=[lv, best_lv]
                    )
                    lv, best_lv = vert_features
                lv.requires_grad = True
                optim = torch.optim.Adam([lv], lr=args.lr)

        t1 = time()

        # Final remesh if requested
        if args.remesh_with_forward_at_end:
            with torch.no_grad():
                search_template, vert_features = remesh_template_from_deformed(
                    pred_mesh, search_template, vert_features=[best_lv]
                )
                best_lv = vert_features[0]
        t2 = time()

        # Generate final mesh
        with torch.no_grad():
            pred_mesh = decoder(search_template, best_lv, expand_lv=False)[-1]
            if args.remesh_at_end:
                pred_mesh = remesh_bk(pred_mesh)
        t3 = time()

        # Store results
        all_pred_meshes.append(pred_mesh.clone().cpu())
        all_filenames.append(mesh_filenames[i])
        all_latent_vectors.append(best_lv.clone().cpu())

        # Save optimized mesh
        mesh_filename = mesh_filenames[i]
        mesh_name = os.path.splitext(mesh_filename)[0]  # Remove .obj extension
        output_mesh_path = join(mesh_output_dir, f'{mesh_name}_optimized.obj')
        save_mesh_to_obj(pred_mesh, output_mesh_path)

        # Compute metrics
        true_point_samples = sample_points_from_meshes(true_mesh, 100000)
        pred_point_samples = sample_points_from_meshes(pred_mesh, 100000)

        metrics = point_metrics(true_point_samples, pred_point_samples, [0.01, 0.02])
        metrics['ChamferL2 x 10000'] = chamfer_distance(true_point_samples, pred_point_samples)[0] * 10000
        metrics['BL quality'] = 1.0 - mesh_bl_quality_loss(pred_mesh)
        
        if METRICS_AVAILABLE:
            metrics['No. ints.'] = 100 * float(self_intersections(pred_mesh.cpu())[0][0]) / len(pred_mesh.faces_packed())
        else:
            metrics['No. ints.'] = 0.0
            
        metrics['Search'] = t1 - t0
        metrics['Remesh'] = t2 - t1
        metrics['Decode'] = t3 - t2
        metrics['Total'] = t2 - t0

        all_metrics.append(metrics)
    
    return all_metrics, all_pred_meshes, all_latent_vectors, all_filenames


def print_metrics_table(metrics, train_test_split_idx, stream=sys.stdout):
    """Print metrics table to stream."""
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


def save_results(all_metrics, all_pred_meshes, all_latent_vectors, all_filenames, args):
    """Save inference results and metrics."""
    # Aggregate metrics
    metrics = defaultdict(list)
    for m in all_metrics:
        for k, v in m.items():
            metrics[k].append(v)
    metrics = {k: torch.tensor(v) for k, v in metrics.items()}

    # Print metrics
    print_metrics_table(metrics, args.train_test_split_idx)
    
    # Save metrics to file
    with open(join(args.output_dir, 'metrics.txt'), 'w') as f:
        print_metrics_table(metrics, args.train_test_split_idx, f)

    # Save arguments
    with open(join(args.output_dir, 'args.txt'), 'w') as f:
        print(args, file=f)

    # Save inference results with automatic suffix if file exists
    base_out_fname = join(args.output_dir, 'inference_results.pt')
    out_fname = base_out_fname
    results_counter = 1
    while os.path.exists(out_fname):
        out_fname = join(args.output_dir, f'inference_results_{results_counter}.pt')
        results_counter += 1

    print(f'Writing results to: {out_fname}')
    torch.save({
        'pred_meshes': all_pred_meshes,
        'metrics': metrics,
        'args': args,
    }, out_fname)

    # Save latent vectors with automatic suffix if file exists
    base_latent_fname = join(args.output_dir, 'latent_vectors.pt')
    latent_fname = base_latent_fname
    latent_counter = 1
    while os.path.exists(latent_fname):
        latent_fname = join(args.output_dir, f'latent_vectors_{latent_counter}.pt')
        latent_counter += 1

    print(f'Writing latent vectors to: {latent_fname}')
    torch.save({
        'filenames': all_filenames,
        'latent_vectors': all_latent_vectors,
        'args': args,
    }, latent_fname)

    # Print summary
    print(f'\nFiles saved:')
    print(f'  - Inference results: {os.path.basename(out_fname)}')
    print(f'  - Latent vectors: {os.path.basename(latent_fname)}')
    print(f'  - Optimized meshes: {len(all_filenames)} meshes saved in meshes/ directory')
    if results_counter > 1 or latent_counter > 1:
        print(f'  - Note: Files were saved with suffixes to avoid overwriting existing files')


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    print(args)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    device, decoder, latent_vectors, template, meshes, mesh_filenames = load_model_and_data(args)
    
    # Process remesh iterations
    remesh_at = process_remesh_iterations(args)
    
    # Run inference based on mode
    if args.latent_mode == 'global':
        all_metrics, all_pred_meshes, all_latent_vectors, all_filenames = run_global_inference(
            device, decoder, latent_vectors, template, meshes, mesh_filenames, args, remesh_at
        )
    elif args.latent_mode == 'local':
        all_metrics, all_pred_meshes, all_latent_vectors, all_filenames = run_local_inference(
            device, decoder, latent_vectors, template, meshes, mesh_filenames, args, remesh_at
        )
    else:
        raise ValueError(f"Unknown latent mode: {args.latent_mode}")
    
    print('\nDone')
    
    # Save results
    save_results(all_metrics, all_pred_meshes, all_latent_vectors, all_filenames, args)
    
    print('All done. Exiting')


if __name__ == '__main__':
    main()
