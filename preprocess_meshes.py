#!/usr/bin/env python3
"""
Comprehensive mesh preprocessing pipeline with 8 different combinations:
- Data split: separate vs mixed
- Scaling: individual vs global  
- Augmentation: noaug vs aug

Creates organized folder structure for all preprocessing combinations.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import trimesh
import shutil
import glob
import re
from pathlib import Path
from time import perf_counter as time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augment.point_wolf import augment_meshes
from util.data import load_meshes_in_dir
from util import seed_everything


class MeshPreprocessor:
    def __init__(self, base_output_dir="/home/ralbe/DALS/mesh_autodecoder/data"):
        self.base_output_dir = base_output_dir
        self.cirrhotic_meshes_folder = '/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_T1_meshes'
        self.healthy_meshes_folder = '/home/ralbe/pyhppc_project/cirr_segm_clean/healthy_T1_meshes'
        self.csv_path = '/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/cirrhotic_dataset.csv'
        
        # Test indices for separate split (from original implementation)
        self.test_indices = [115, 272, 126, 91, 195, 314, 19, 131, 389, 500, 482, 152, 153, 149, 422, 72, 124, 150, 355, 51, 511, 323, 293, 20, 427, 172, 332, 64, 191, 381, 294, 433, 392, 158, 62, 351, 296, 146, 245, 10, 259, 287, 431, 292, 114, 193, 463, 487, 455, 408, 290, 337, 8, 192, 394, 68, 501, 168, 458, 448, 198, 160, 386, 129, 362, 31, 361, 80, 9, 316, 460, 291, 250, 200, 252, 24, 208, 319, 154, 382, 249, 400, 98, 488, 492, 263, 377, 196, 254, 128, 25, 65, 37, 42, 171, 96, 88, 125, 335, 73, 352, 366, 157, 446, 106, 378, 184, 311, 447, 82, 81, 21, 137, 365, 282, 17, 142, 338, 174, 175, 283, 141, 89, 103, 410, 140, 425, 253, 209, 289, 162, 434, 261, 437, 274, 396, 26, 360, 489, 353, 213, 105, 38, 490, 278, 312, 199, 481, 50, 464, 273, 452]
        
        # Load CSV data for mixed split
        self.df = pd.read_csv(self.csv_path)
        self.patient_severity_map = dict(zip(self.df['Patient ID'], self.df['Radiological Evaluation']))
        
        print(f"Loaded CSV with {len(self.df)} patients")
        print(f"Severity distribution: {self.df['Radiological Evaluation'].value_counts().to_dict()}")
    
    def create_folder_structure(self):
        """Create the 8 preprocessing folder combinations"""
        combinations = [
            "separate_individual_noaug",
            "separate_individual_aug", 
            "separate_global_noaug",
            "separate_global_aug",
            "mixed_individual_noaug",
            "mixed_individual_aug",
            "mixed_global_noaug", 
            "mixed_global_aug"
        ]
        
        for combo in combinations:
            combo_dir = os.path.join(self.base_output_dir, combo)
            os.makedirs(combo_dir, exist_ok=True)
            
            # Create train/val/test subdirectories
            for split in ['train_meshes', 'val_meshes', 'test_meshes']:
                os.makedirs(os.path.join(combo_dir, split), exist_ok=True)
        
        print(f"Created folder structure for {len(combinations)} preprocessing combinations")
        return combinations
    
    def extract_patient_id_from_filename(self, filename):
        """Extract patient ID from mesh filename"""
        # Extract number from filename (e.g., "cirrhotic_123.obj" -> 123)
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return None
    
    def get_severity_for_mesh(self, filename):
        """Get disease severity for a mesh file"""
        patient_id = self.extract_patient_id_from_filename(filename)
        if patient_id is None:
            return None
        return self.patient_severity_map.get(patient_id, None)
    
    def separate_split(self, cirrhotic_meshes, healthy_meshes):
        """Separate split: healthy->train, cirrhotic->val/test based on test_indices"""
        print("Performing separate split...")
        
        # All healthy meshes go to training
        train_meshes = healthy_meshes.copy()
        
        # Split cirrhotic meshes based on test_indices
        test_meshes = []
        val_meshes = []
        
        for mesh_file in cirrhotic_meshes:
            filename = os.path.basename(mesh_file)
            patient_id = self.extract_patient_id_from_filename(filename)
            
            if patient_id in self.test_indices:
                test_meshes.append(mesh_file)
            else:
                val_meshes.append(mesh_file)
        
        print(f"Separate split: {len(train_meshes)} train (healthy), {len(val_meshes)} val (cirrhotic), {len(test_meshes)} test (cirrhotic)")
        return train_meshes, val_meshes, test_meshes
    
    def mixed_split(self, cirrhotic_meshes, healthy_meshes):
        """Mixed split: stratified 70/15/15 based on disease severity"""
        print("Performing mixed stratified split...")
        
        # Prepare data for stratification
        all_meshes = []
        severities = []
        
        # Add healthy meshes (severity = 0)
        for mesh_file in healthy_meshes:
            all_meshes.append(mesh_file)
            severities.append(0)  # Healthy patients have severity 0
        
        # Add cirrhotic meshes with their severity
        for mesh_file in cirrhotic_meshes:
            filename = os.path.basename(mesh_file)
            severity = self.get_severity_for_mesh(filename)
            if severity is not None:
                all_meshes.append(mesh_file)
                severities.append(severity)
            else:
                print(f"Warning: Could not find severity for {filename}, skipping")
        
        # Convert to numpy arrays
        all_meshes = np.array(all_meshes)
        severities = np.array(severities)
        
        print(f"Severity distribution before split: {np.bincount(severities)}")
        
        # First split: 70% train, 30% temp
        train_meshes, temp_meshes, train_sev, temp_sev = train_test_split(
            all_meshes, severities, test_size=0.3, stratify=severities, random_state=42
        )
        
        # Second split: 15% val, 15% test from temp
        val_meshes, test_meshes, val_sev, test_sev = train_test_split(
            temp_meshes, temp_sev, test_size=0.5, stratify=temp_sev, random_state=42
        )
        
        print(f"Mixed split: {len(train_meshes)} train, {len(val_meshes)} val, {len(test_meshes)} test")
        print(f"Train severity distribution: {np.bincount(train_sev)}")
        print(f"Val severity distribution: {np.bincount(val_sev)}")
        print(f"Test severity distribution: {np.bincount(test_sev)}")
        
        return train_meshes.tolist(), val_meshes.tolist(), test_meshes.tolist()
    
    def find_global_max_radius(self, mesh_files):
        """Find global maximum radius across all meshes"""
        print("Finding global maximum radius...")
        global_max_radius = 0.0
        
        for mesh_file in tqdm(mesh_files, desc="Computing radii"):
            try:
                mesh = trimesh.load(mesh_file)
                verts = mesh.vertices
                verts_centered = verts - verts.mean(axis=0)
                max_radius = (verts_centered ** 2).sum(axis=1).max() ** 0.5
                global_max_radius = max(global_max_radius, max_radius)
            except Exception as e:
                print(f"Error processing {mesh_file}: {e}")
                continue
        
        print(f"Global maximum radius: {global_max_radius:.6f}")
        return global_max_radius
    
    def scale_mesh_individual(self, mesh_file, output_path):
        """Scale mesh individually to unit sphere"""
        mesh = trimesh.load(mesh_file)
        verts = mesh.vertices
        
        # Center the mesh
        verts_centered = verts - verts.mean(axis=0)
        
        # Scale to unit sphere (max radius = 1.0)
        max_dist = (verts_centered ** 2).sum(axis=1).max() ** 0.5
        verts_scaled = verts_centered / max_dist
        
        # Update mesh vertices
        mesh.vertices = verts_scaled
        mesh.export(output_path)
    
    def scale_mesh_global(self, mesh_file, output_path, global_max_radius):
        """Scale mesh using global maximum radius"""
        mesh = trimesh.load(mesh_file)
        verts = mesh.vertices
        
        # Center the mesh
        verts_centered = verts - verts.mean(axis=0)
        
        # Scale using global maximum radius
        verts_scaled = verts_centered / global_max_radius
        
        # Update mesh vertices
        mesh.vertices = verts_scaled
        mesh.export(output_path)
    
    def get_mesh_prefix(self, mesh_file):
        """Get prefix to identify if mesh is from healthy or cirrhotic data"""
        if 'healthy' in mesh_file.lower():
            return 'healthy'
        elif 'cirrhotic' in mesh_file.lower():
            return 'cirrhotic'
        else:
            return 'unknown'
    
    def augment_meshes_individual(self, mesh_files, output_dir, num_augment=99):
        """Augment meshes individually to ensure correct count"""
        print(f"Augmenting {len(mesh_files)} meshes with {num_augment} variants each...")
        
        successful_count = 0
        failed_count = 0
        
        for i, mesh_file in enumerate(mesh_files):
            try:
                # Load single mesh
                mesh = trimesh.load(mesh_file)
                base_name = os.path.splitext(os.path.basename(mesh_file))[0]
                prefix = self.get_mesh_prefix(mesh_file)
                
                # Convert to PyTorch3D format
                from pytorch3d.structures import Meshes
                import torch
                
                verts = torch.tensor(mesh.vertices, dtype=torch.float32)
                faces = torch.tensor(mesh.faces, dtype=torch.long)
                pytorch3d_mesh = Meshes([verts], [faces])
                
                # Augment this single mesh
                augmented_meshes = augment_meshes(
                    pytorch3d_mesh,  # List containing single mesh
                    num_augment=num_augment,
                    num_anchor=4,
                    sample_type='fps',
                    sigma=0.8,
                    R_range=25,
                    S_range=2.0,
                    T_range=2.5,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                # Save original mesh with prefix
                orig_path = os.path.join(output_dir, f"{prefix}_{base_name}_orig.obj")
                mesh.export(orig_path)
                
                # Save augmented variants for this mesh
                aug_count = 0
                
                for j, aug_meshes_batch in enumerate(augmented_meshes):
                    aug_path = os.path.join(output_dir, f"{prefix}_{base_name}_aug_{j:03d}.obj")
                    
                    # Convert back to trimesh format
                    verts_np = aug_meshes_batch.verts_packed().detach().cpu().numpy()
                    faces_np = aug_meshes_batch.faces_packed().detach().cpu().numpy()
                    
                    aug_trimesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
                    aug_trimesh.export(aug_path)
                    aug_count += 1
                
                print(f"  Augmented {prefix}_{base_name}: 1 original + {aug_count} variants")
                successful_count += 1
                
            except Exception as e:
                print(f"Error augmenting {mesh_file}: {e}")
                failed_count += 1
                # Still save the original mesh even if augmentation fails
                try:
                    mesh = trimesh.load(mesh_file)
                    base_name = os.path.splitext(os.path.basename(mesh_file))[0]
                    prefix = self.get_mesh_prefix(mesh_file)
                    orig_path = os.path.join(output_dir, f"{prefix}_{base_name}_orig.obj")
                    mesh.export(orig_path)
                    print(f"  Saved original only for {prefix}_{base_name} (augmentation failed)")
                except Exception as e2:
                    print(f"  Failed to save original for {mesh_file}: {e2}")
                continue
        
        print(f"Augmentation complete: {successful_count} successful, {failed_count} failed")
        return successful_count, failed_count
    
    def process_combination(self, split_type, scaling_type, augmentation_type, num_augment=99):
        """Process a single preprocessing combination"""
        combo_name = f"{split_type}_{scaling_type}_{augmentation_type}"
        combo_dir = os.path.join(self.base_output_dir, combo_name)
        
        print(f"\n{'='*60}")
        print(f"Processing combination: {combo_name}")
        print(f"{'='*60}")
        
        # Clean up existing directory to avoid leftover files
        if os.path.exists(combo_dir):
            print(f"Removing existing directory: {combo_dir}")
            shutil.rmtree(combo_dir)
        
        # Recreate directory structure
        os.makedirs(combo_dir, exist_ok=True)
        for split in ['train_meshes', 'val_meshes', 'test_meshes']:
            os.makedirs(os.path.join(combo_dir, split), exist_ok=True)
        
        # Load all meshes
        print("Loading meshes...")
        cirrhotic_meshes = glob.glob(os.path.join(self.cirrhotic_meshes_folder, "*.obj"))
        healthy_meshes = glob.glob(os.path.join(self.healthy_meshes_folder, "*.obj"))
        
        print(f"Found {len(cirrhotic_meshes)} cirrhotic meshes")
        print(f"Found {len(healthy_meshes)} healthy meshes")
        
        # Perform data split
        if split_type == "separate":
            train_meshes, val_meshes, test_meshes = self.separate_split(cirrhotic_meshes, healthy_meshes)
        else:  # mixed
            train_meshes, val_meshes, test_meshes = self.mixed_split(cirrhotic_meshes, healthy_meshes)
        
        # Find global max radius if needed
        global_max_radius = None
        if scaling_type == "global":
            all_meshes = train_meshes + val_meshes + test_meshes
            global_max_radius = self.find_global_max_radius(all_meshes)
        
        # Process each split
        for split_name, mesh_files in [("train_meshes", train_meshes), 
                                      ("val_meshes", val_meshes), 
                                      ("test_meshes", test_meshes)]:
            
            split_dir = os.path.join(combo_dir, split_name)
            print(f"\nProcessing {split_name}: {len(mesh_files)} meshes")
            
            if augmentation_type == "aug" and split_name == "train_meshes":
                # Apply augmentation to training meshes
                self.augment_meshes_individual(mesh_files, split_dir, num_augment=num_augment)
            else:
                # Copy and scale meshes without augmentation
                processed_count = 0
                for mesh_file in tqdm(mesh_files, desc=f"Processing {split_name}"):
                    try:
                        filename = os.path.basename(mesh_file)
                        prefix = self.get_mesh_prefix(mesh_file)
                        base_name = os.path.splitext(filename)[0]
                        output_filename = f"{prefix}_{base_name}.obj"
                        output_path = os.path.join(split_dir, output_filename)
                        
                        if scaling_type == "individual":
                            self.scale_mesh_individual(mesh_file, output_path)
                        else:  # global
                            self.scale_mesh_global(mesh_file, output_path, global_max_radius)
                        
                        processed_count += 1
                            
                    except Exception as e:
                        print(f"Error processing {mesh_file}: {e}")
                        continue
                
                print(f"  Successfully processed {processed_count}/{len(mesh_files)} meshes")
        
        print(f"Completed processing {combo_name}")
    
    def run_all_combinations(self, num_augment=99):
        """Run all 8 preprocessing combinations"""
        print("Starting comprehensive mesh preprocessing pipeline...")
        print(f"Using {num_augment} augmented variants per original mesh")
        
        # Create folder structure
        combinations = self.create_folder_structure()
        
        # Process each combination
        for combo in combinations:
            parts = combo.split('_')
            split_type = parts[0]      # separate or mixed
            scaling_type = parts[1]    # individual or global
            augmentation_type = parts[2]  # noaug or aug
            
            self.process_combination(split_type, scaling_type, augmentation_type, num_augment=num_augment)
        
        print(f"\n{'='*60}")
        print("PREPROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"Created {len(combinations)} preprocessing combinations in {self.base_output_dir}")
        
        # Print summary
        for combo in combinations:
            combo_dir = os.path.join(self.base_output_dir, combo)
            print(f"\n{combo}:")
            for split in ['train_meshes', 'val_meshes', 'test_meshes']:
                split_dir = os.path.join(combo_dir, split)
                if os.path.exists(split_dir):
                    # Count all .obj files in the directory
                    total_count = len(glob.glob(os.path.join(split_dir, "*.obj")))
                    print(f"  {split}: {total_count} meshes")
                    
                    # For augmented training sets, show breakdown
                    if split == 'train_meshes' and 'aug' in combo:
                        orig_count = len(glob.glob(os.path.join(split_dir, "*_orig.obj")))
                        aug_count = len(glob.glob(os.path.join(split_dir, "*_aug_*.obj")))
                        print(f"    -> {orig_count} original + {aug_count} augmented = {total_count} total")
                    else:
                        # For non-augmented sets, show healthy/cirrhotic breakdown
                        healthy_count = len(glob.glob(os.path.join(split_dir, "healthy_*.obj")))
                        cirrhotic_count = len(glob.glob(os.path.join(split_dir, "cirrhotic_*.obj")))
                        if healthy_count > 0 or cirrhotic_count > 0:
                            print(f"    -> {healthy_count} healthy + {cirrhotic_count} cirrhotic = {total_count} total")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive mesh preprocessing pipeline")
    parser.add_argument('--base_output_dir', type=str, 
                       default='/home/ralbe/DALS/mesh_autodecoder/data',
                       help='Base directory for all preprocessing outputs')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--num_augment', type=int, default=9,
                       help='Number of augmented variants to create per original mesh (default: 9)')
    
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.random_seed)
    
    # Create preprocessor and run all combinations
    preprocessor = MeshPreprocessor(base_output_dir=args.base_output_dir)
    preprocessor.run_all_combinations(num_augment=args.num_augment)


if __name__ == '__main__':
    main()
