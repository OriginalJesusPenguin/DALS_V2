#!/usr/bin/env python3
"""
Script to pre-augment meshes and save them to disk for memory-efficient training.
This avoids loading all 36,500+ augmented meshes into memory at once.
"""

import os
import sys
import argparse
from pathlib import Path
from time import perf_counter as time
from tqdm import tqdm

import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augment.point_wolf import augment_meshes
from util.data import load_meshes_in_dir
from util import seed_everything


def save_mesh_to_obj(mesh, filepath):
    """Save a single mesh to OBJ file"""
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    
    # Convert to numpy for saving
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    
    # Save as OBJ file
    with open(filepath, 'w') as f:
        # Write vertices
        for v in verts_np:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-based indexing)
        for face in faces_np:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def normalize_mesh_unit_sphere(mesh: Meshes) -> Meshes:
    """Center and scale a single mesh to unit sphere (max radius ~= 1).

    If max radius is extremely small, returns the mesh unchanged.
    """
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()

    # Center
    verts_centered = verts - verts.mean(dim=0, keepdim=True)

    # Scale by max norm
    norms = torch.sqrt((verts_centered ** 2).sum(1))
    max_norm = norms.max()
    if max_norm.item() < 1e-8:
        return mesh
    scale = (1.0 / max_norm) * 0.999999
    verts_scaled = verts_centered * scale

    return Meshes([verts_scaled], [faces])


def augment_and_save_meshes(
    input_dir,
    output_dir,
    num_augment=100,
    batch_size=10,  # Process this many meshes at a time
    **augment_params
):
    """
    Augment meshes in batches and save to disk to avoid memory issues.
    
    Args:
        input_dir: Directory containing original .obj files
        output_dir: Directory to save augmented meshes
        num_augment: Number of augmentations per mesh
        batch_size: Number of meshes to process at once
        **augment_params: PointWOLF augmentation parameters
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all original meshes
    print(f"Loading meshes from {input_dir}...")
    t_load = time()
    original_meshes = load_meshes_in_dir(input_dir)
    t_load = time() - t_load
    print(f"Loaded {len(original_meshes)} meshes in {t_load:.2f} seconds")
    
    # Get list of original mesh filenames for naming
    input_path = Path(input_dir)
    original_files = sorted([f for f in input_path.glob("*.obj")])
    
    print(f"Starting augmentation with {num_augment} augmentations per mesh...")
    print(f"Processing in batches of {batch_size} meshes...")
    
    total_processed = 0
    total_augmented = 0
    
    # Process meshes in batches
    for batch_start in tqdm(range(0, len(original_meshes), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(original_meshes))
        batch_meshes = original_meshes[batch_start:batch_end]
        batch_files = original_files[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}: meshes {batch_start}-{batch_end-1}")
        
        # Normalize to unit sphere before augmentation to match kernel scale
        normalized_batch_meshes = [normalize_mesh_unit_sphere(m) for m in batch_meshes]

        # Augment this batch
        t_aug = time()
        augmented_meshes = augment_meshes(
            normalized_batch_meshes,
            num_augment=num_augment,
            **augment_params
        )
        t_aug = time() - t_aug
        
        # Save original meshes (first in each group)
        for i, (mesh, orig_file) in enumerate(zip(batch_meshes, batch_files)):
            # Save original mesh
            orig_name = orig_file.stem
            orig_output_path = os.path.join(output_dir, f"{orig_name}_orig.obj")
            save_mesh_to_obj(mesh, orig_output_path)
            total_processed += 1
            
            # Save augmented meshes for this original mesh
            for j in range(num_augment):
                aug_idx = i * num_augment + j
                aug_mesh = augmented_meshes[aug_idx]
                aug_output_path = os.path.join(output_dir, f"{orig_name}_aug_{j:03d}.obj")
                save_mesh_to_obj(aug_mesh, aug_output_path)
                total_augmented += 1
        
        print(f"  Augmented batch in {t_aug:.2f} seconds")
        print(f"  Saved {len(batch_meshes)} original + {len(augmented_meshes)} augmented meshes")
        
        # Clear memory
        del augmented_meshes
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n=== Augmentation Complete ===")
    print(f"Total original meshes processed: {total_processed}")
    print(f"Total augmented meshes created: {total_augmented}")
    print(f"Total meshes in output directory: {total_processed + total_augmented}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pre-augment meshes and save to disk")
    
    # Required arguments
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing original .obj files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save augmented meshes')
    
    # Augmentation parameters
    parser.add_argument('--num_augment', type=int, default=100,
                       help='Number of augmentations per mesh')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Number of meshes to process at once')
    
    # PointWOLF parameters
    parser.add_argument('--pw_num_anchor', type=int, default=4)
    parser.add_argument('--pw_sample_type', type=str, default='fps')
    # Use a much smaller sigma assuming inputs are normalized before augmentation
    parser.add_argument('--pw_sigma', type=float, default=0.05)
    parser.add_argument('--pw_r_range', type=float, default=1)
    parser.add_argument('--pw_s_range', type=float, default=2)
    parser.add_argument('--pw_t_range', type=float, default=0.25)
    
    # Other parameters
    parser.add_argument('--data_random_seed', type=int, default=1337)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    seed_everything(args.data_random_seed)
    
    # Prepare augmentation parameters
    augment_params = {
        'num_anchor': args.pw_num_anchor,
        'sample_type': args.pw_sample_type,
        'sigma': args.pw_sigma,
        'R_range': args.pw_r_range,
        'S_range': args.pw_s_range,
        'T_range': args.pw_t_range,
        'device': args.device,
    }
    
    # Run augmentation
    augment_and_save_meshes(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_augment=args.num_augment,
        batch_size=args.batch_size,
        **augment_params
    )


if __name__ == '__main__':
    main()
