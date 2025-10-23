#!/usr/bin/env python3
"""
Simplified mesh preprocessing pipeline that:
1. Loads T1_mask files from CSV using nibabel
2. Converts to meshes using marching cubes
3. Targets ~2500 vertices per mesh
4. Saves meshes in organized folders for healthy and cirrhotic data
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
import trimesh
from skimage import measure
from tqdm import tqdm
import glob
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import seed_everything


class SimplifiedMeshPreprocessor:
    def __init__(self, base_output_dir="/home/ralbe/DALS/mesh_autodecoder/simplified_meshes"):
        self.base_output_dir = base_output_dir
        self.cirrhotic_csv_path = '/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/cirrhotic_dataset.csv'
        self.healthy_csv_path = '/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/healthy_dataset.csv'
        self.target_vertices = 2500
        
        # Create output directories
        self.cirrhotic_output_dir = os.path.join(base_output_dir, 'cirrhotic_meshes')
        self.healthy_output_dir = os.path.join(base_output_dir, 'healthy_meshes')
        
        os.makedirs(self.cirrhotic_output_dir, exist_ok=True)
        os.makedirs(self.healthy_output_dir, exist_ok=True)
        
        print(f"Output directories created:")
        print(f"  Cirrhotic: {self.cirrhotic_output_dir}")
        print(f"  Healthy: {self.healthy_output_dir}")
    
    def load_nifti_mask(self, nifti_path):
        """Load NIfTI mask file using nibabel"""
        try:
            nifti_img = nib.load(nifti_path)
            mask_data = nifti_img.get_fdata()
            return mask_data, nifti_img.affine
        except Exception as e:
            print(f"Error loading {nifti_path}: {e}")
            return None, None
    
    def marching_cubes_to_mesh(self, mask_data, level=0.5):
        """Convert binary mask to mesh using marching cubes"""
        try:
            # Apply marching cubes
            verts, faces, _, _ = measure.marching_cubes(mask_data, level=level)
            return verts, faces
        except Exception as e:
            print(f"Error in marching cubes: {e}")
            return None, None
    
    def decimate_mesh_to_target(self, verts, faces, target_vertices):
        """Decimate mesh to approximately target number of vertices using progressive face removal"""
        try:
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            
            if len(mesh.vertices) > target_vertices:
                # Iteratively remove faces with smallest area until close to target vertex count
                while len(mesh.vertices) > target_vertices and len(mesh.faces) > 0:
                    # Compute face areas
                    face_areas = mesh.area_faces
                    # Find face with smallest area
                    smallest_face_idx = face_areas.argmin()
                    # Remove that face
                    mask = np.ones(len(mesh.faces), dtype=bool)
                    mask[smallest_face_idx] = False
                    mesh.update_faces(mask)
                    mesh.remove_unreferenced_vertices()
                    # Break if mesh becomes too small or non-watertight
                    if len(mesh.vertices) <= target_vertices or len(mesh.faces) < 10:
                        break

            return mesh.vertices, mesh.faces
        except Exception as e:
            print(f"Error in mesh decimation: {e}")
            return verts, faces
    
    def center_and_scale_mesh(self, verts):
        """Center mesh and scale to unit sphere"""
        # Center the mesh
        verts_centered = verts - verts.mean(axis=0)
        
        # Scale to unit sphere (max radius = 1.0)
        max_dist = (verts_centered ** 2).sum(axis=1).max() ** 0.5
        if max_dist > 0:
            verts_scaled = verts_centered / max_dist
        else:
            verts_scaled = verts_centered
            
        return verts_scaled
    
    def process_single_mask(self, nifti_path, output_path, patient_id, data_type):
        """Process a single T1 mask file to mesh"""
        print(f"Processing {data_type} patient {patient_id}: {os.path.basename(nifti_path)}")
        
        # Load NIfTI mask
        mask_data, affine = self.load_nifti_mask(nifti_path)
        if mask_data is None:
            return False
        
        # Convert to mesh using marching cubes
        verts, faces = self.marching_cubes_to_mesh(mask_data)
        if verts is None or faces is None:
            return False
        
        # Decimate to target vertices
        verts_decimated, faces_decimated = self.decimate_mesh_to_target(verts, faces, self.target_vertices)
        
        # Center and scale mesh
        verts_final = self.center_and_scale_mesh(verts_decimated)
        
        # Create final mesh
        mesh = trimesh.Trimesh(vertices=verts_final, faces=faces_decimated)
        
        # Save mesh
        mesh.export(output_path)
        
        print(f"  -> Saved mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return True
    
    def process_dataset(self, csv_path, output_dir, data_type):
        """Process all masks in a dataset"""
        print(f"\n{'='*60}")
        print(f"Processing {data_type} dataset")
        print(f"{'='*60}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} {data_type} patients")
        
        # Get T1_mask column
        if 'T1_mask' not in df.columns:
            print(f"Error: T1_mask column not found in {csv_path}")
            return
        
        successful_count = 0
        failed_count = 0
        
        # Process each patient
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {data_type}"):
            patient_id = row['Patient ID']
            t1_mask_path = row['T1_mask']
            
            # Check if file exists
            if not os.path.exists(t1_mask_path):
                print(f"Warning: File not found: {t1_mask_path}")
                failed_count += 1
                continue
            
            # Create output filename
            output_filename = f"{data_type}_{patient_id}.obj"
            output_path = os.path.join(output_dir, output_filename)
            
            # Process the mask
            success = self.process_single_mask(t1_mask_path, output_path, patient_id, data_type)
            
            if success:
                successful_count += 1
            else:
                failed_count += 1
        
        print(f"\n{data_type} processing complete:")
        print(f"  Successful: {successful_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {successful_count + failed_count}")
    
    def run_all_processing(self):
        """Process both healthy and cirrhotic datasets"""
        print("Starting simplified mesh preprocessing pipeline...")
        print(f"Target vertices per mesh: {self.target_vertices}")
        
        # Process cirrhotic dataset
        self.process_dataset(self.cirrhotic_csv_path, self.cirrhotic_output_dir, "cirrhotic")
        
        # Process healthy dataset
        self.process_dataset(self.healthy_csv_path, self.healthy_output_dir, "healthy")
        
        # Print final summary
        print(f"\n{'='*60}")
        print("PREPROCESSING COMPLETE!")
        print(f"{'='*60}")
        
        # Count final meshes
        cirrhotic_count = len(glob.glob(os.path.join(self.cirrhotic_output_dir, "*.obj")))
        healthy_count = len(glob.glob(os.path.join(self.healthy_output_dir, "*.obj")))
        
        print(f"Final mesh counts:")
        print(f"  Cirrhotic meshes: {cirrhotic_count}")
        print(f"  Healthy meshes: {healthy_count}")
        print(f"  Total meshes: {cirrhotic_count + healthy_count}")
        print(f"\nOutput directory: {self.base_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Simplified mesh preprocessing pipeline")
    parser.add_argument('--base_output_dir', type=str, 
                       default='/home/ralbe/DALS/mesh_autodecoder/simplified_meshes',
                       help='Base directory for output meshes')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--target_vertices', type=int, default=2500,
                       help='Target number of vertices per mesh (default: 2500)')
    
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.random_seed)
    
    # Create preprocessor and run processing
    preprocessor = SimplifiedMeshPreprocessor(
        base_output_dir=args.base_output_dir
    )
    preprocessor.target_vertices = args.target_vertices
    preprocessor.run_all_processing()


if __name__ == '__main__':
    main()
