#!/usr/bin/env python3
"""
Prepare the full mixed dataset (healthy + cirrhotic) for training.
This script processes both healthy and cirrhotic data using existing train/valid/test splits.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from prepare_cirrhotic_data import load_nifti_volume
from sklearn.model_selection import train_test_split
import gc
from tqdm import tqdm

def create_full_dataset(target_size=(192, 192, 192)):
    """
    Create the full mixed dataset with healthy and cirrhotic samples.
    Uses existing train/valid/test splits from cirrhotic data and appends healthy data split 70/15/15.
    
    Args:
        target_size: Target volume size for resizing
    """
    
    # Define CSV paths
    cirrhotic_csv = '/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/cirrhotic_dataset.csv'
    healthy_csv = '/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/healthy_dataset.csv'
    
    print("Creating mixed dataset...")
    print("Using existing cirrhotic splits + 70/15/15 healthy splits")
    
    # Load CSV files
    cirrhotic_df = pd.read_csv(cirrhotic_csv)
    healthy_df = pd.read_csv(healthy_csv)
    
    print(f"Cirrhotic samples: {len(cirrhotic_df)}")
    print(f"Healthy samples: {len(healthy_df)}")
    
    # Get cirrhotic data by group (using existing group column)
    def get_cirrhotic_data_by_group(df):
        groups = {}
        for group in ['train', 'valid', 'test']:
            group_data = df[df['group'] == group]
            groups[group] = group_data
            print(f"Cirrhotic {group}: {len(group_data)} samples")
        return groups
    
    # Create healthy data splits (70/15/15)
    def create_healthy_splits(df, random_state=42):
        # First split: 70% train, 30% temp
        train_df, temp_df = train_test_split(df, train_size=0.7, random_state=random_state)
        # Second split: 15% valid, 15% test from temp (50/50 split of the 30%)
        valid_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=random_state)
        
        groups = {
            'train': train_df,
            'valid': valid_df, 
            'test': test_df
        }
        
        print(f"Healthy train: {len(train_df)} samples")
        print(f"Healthy valid: {len(valid_df)} samples")
        print(f"Healthy test: {len(test_df)} samples")
        
        return groups
    
    cirrhotic_groups = get_cirrhotic_data_by_group(cirrhotic_df)
    healthy_groups = create_healthy_splits(healthy_df)
    
    # Use all cirrhotic data as-is (with existing splits)
    # Split healthy data 70/15/15 and append to corresponding cirrhotic splits
    print(f"\nDataset composition:")
    print(f"Cirrhotic train: {len(cirrhotic_groups['train'])} samples")
    print(f"Cirrhotic valid: {len(cirrhotic_groups['valid'])} samples") 
    print(f"Cirrhotic test: {len(cirrhotic_groups['test'])} samples")
    print(f"Healthy train: {len(healthy_groups['train'])} samples")
    print(f"Healthy valid: {len(healthy_groups['valid'])} samples")
    print(f"Healthy test: {len(healthy_groups['test'])} samples")
    
    print(f"\nFinal dataset distribution:")
    print(f"Train: {len(cirrhotic_groups['train'])} cirrhotic + {len(healthy_groups['train'])} healthy = {len(cirrhotic_groups['train']) + len(healthy_groups['train'])} total")
    print(f"Valid: {len(cirrhotic_groups['valid'])} cirrhotic + {len(healthy_groups['valid'])} healthy = {len(cirrhotic_groups['valid']) + len(healthy_groups['valid'])} total")
    print(f"Test: {len(cirrhotic_groups['test'])} cirrhotic + {len(healthy_groups['test'])} healthy = {len(cirrhotic_groups['test']) + len(healthy_groups['test'])} total")
    
    # Check available files before processing
    def check_available_files(df, data_type):
        """Check which files are actually available"""
        available_count = 0
        missing_files = []
        
        for idx, row in df.iterrows():
            img_path = row['T1_img']
            gt_path = row['T1_mask']
            pred_path = row['T1_mask_AttentionUNet']
            
            if (pd.isna(img_path) or not os.path.exists(img_path) or
                pd.isna(gt_path) or not os.path.exists(gt_path) or
                pd.isna(pred_path) or not os.path.exists(pred_path)):
                missing_files.append(row['Patient ID'])
            else:
                available_count += 1
        
        print(f"{data_type} available files: {available_count}/{len(df)}")
        if missing_files:
            print(f"{data_type} missing files: {len(missing_files)} patients")
        return available_count
    
    # Check available files
    cirrhotic_train_available = check_available_files(cirrhotic_groups['train'], "Cirrhotic train")
    cirrhotic_valid_available = check_available_files(cirrhotic_groups['valid'], "Cirrhotic valid")
    cirrhotic_test_available = check_available_files(cirrhotic_groups['test'], "Cirrhotic test")
    healthy_train_available = check_available_files(healthy_groups['train'], "Healthy train")
    healthy_valid_available = check_available_files(healthy_groups['valid'], "Healthy valid")
    healthy_test_available = check_available_files(healthy_groups['test'], "Healthy test")
    
    print(f"\nAvailable files:")
    print(f"Train: {cirrhotic_train_available} cirrhotic + {healthy_train_available} healthy = {cirrhotic_train_available + healthy_train_available} total")
    print(f"Valid: {cirrhotic_valid_available} cirrhotic + {healthy_valid_available} healthy = {cirrhotic_valid_available + healthy_valid_available} total")
    print(f"Test: {cirrhotic_test_available} cirrhotic + {healthy_test_available} healthy = {cirrhotic_test_available + healthy_test_available} total")
    
    def process_and_save_data(df, data_type, target_size, output_path, batch_size=3):
        """Process patient data and save incrementally to avoid memory issues"""
        # Filter out patients with missing files first
        valid_patients = []
        for idx, row in df.iterrows():
            img_path = row['T1_img']
            gt_path = row['T1_mask']
            pred_path = row['T1_mask']
            
            if (not pd.isna(img_path) and os.path.exists(img_path) and
                not pd.isna(gt_path) and os.path.exists(gt_path) and
                not pd.isna(pred_path) and os.path.exists(pred_path)):
                valid_patients.append(row)
        
        print(f"Processing {len(valid_patients)} valid {data_type} patients in batches of {batch_size}")
        
        # Initialize lists to store data
        all_images = []
        all_labels = []
        all_masks = []
        temp_file_count = 0
        
        # Process in batches and save incrementally
        for i in range(0, len(valid_patients), batch_size):
            batch_patients = valid_patients[i:i+batch_size]
            batch_images = []
            batch_labels = []
            batch_masks = []
            
            for patient in tqdm(batch_patients, desc=f"Processing {data_type} batch {i//batch_size + 1}/{(len(valid_patients)-1)//batch_size + 1}", leave=False):
                try:
                    # Load volumes
                    image = load_nifti_volume(patient['T1_img'], target_size)
                    label = load_nifti_volume(patient['T1_mask'], target_size)
                    mask = load_nifti_volume(patient['T1_mask_AttentionUNet'], target_size)
                    
                    if image is None or label is None or mask is None:
                        print(f"Warning: Failed to load data for {data_type} patient {patient['Patient ID']}")
                        continue
                    
                    # Convert to tensors
                    image = torch.from_numpy(image).float()
                    label = torch.from_numpy(label).float()
                    mask = torch.from_numpy(mask).float()
                    
                    batch_images.append(image)
                    batch_labels.append(label)
                    batch_masks.append(mask)
                    
                except Exception as e:
                    print(f"Error processing {data_type} patient {patient['Patient ID']}: {e}")
                    continue
            
            # Add batch to all data
            all_images.extend(batch_images)
            all_labels.extend(batch_labels)
            all_masks.extend(batch_masks)
        
        # Clear batch data from memory
            del batch_images, batch_labels, batch_masks
            gc.collect()
            
            # Save incrementally every 20 samples to avoid memory buildup
            if len(all_images) >= 20:
                print(f"Saving intermediate data: {len(all_images)} samples...")
                temp_data = {
                    'images': torch.stack(all_images),
                    'labels': torch.stack(all_labels),
                    'masks': torch.stack(all_masks)
                }
                
                # Save to unique temporary file
                temp_path = output_path.replace('.pt', f'_temp_{temp_file_count}.pt')
                torch.save(temp_data, temp_path)
                temp_file_count += 1
                
                # Clear memory
                del temp_data, all_images, all_labels, all_masks
                gc.collect()
                
                # Reinitialize lists
                all_images = []
                all_labels = []
                all_masks = []
        
        # Save any remaining data
        if len(all_images) > 0:
            print(f"Saving final data: {len(all_images)} samples...")
            temp_data = {
                'images': torch.stack(all_images),
                'labels': torch.stack(all_labels),
                'masks': torch.stack(all_masks)
            }
            temp_path = output_path.replace('.pt', f'_temp_{temp_file_count}.pt')
            torch.save(temp_data, temp_path)
            temp_file_count += 1
            del temp_data, all_images, all_labels, all_masks
        gc.collect()
        
        # Return list of all temp files created
        temp_files = []
        for i in range(temp_file_count):
            temp_path = output_path.replace('.pt', f'_temp_{i}.pt')
            if os.path.exists(temp_path):
                temp_files.append(temp_path)
        
        return temp_files
    
    def combine_temp_files(temp_files, final_path):
        """Combine temporary files into final dataset using memory-efficient approach"""
        print(f"Combining {len(temp_files)} temporary files...")
        
        if not temp_files:
            print("No temporary files to combine!")
            return None
        
        # Load first file to get shape info
        first_data = torch.load(temp_files[0])
        sample_shape = first_data['images'].shape[1:]  # Get shape without batch dimension
        
        print(f"Sample shape: {sample_shape}")
        print(f"Total samples: {sum([torch.load(f)['images'].shape[0] for f in temp_files])}")
        
        # Calculate total samples
        total_samples = 0
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                temp_data = torch.load(temp_file)
                total_samples += temp_data['images'].shape[0]
                del temp_data
                gc.collect()
        
        print(f"Creating final tensors for {total_samples} samples...")
        
        # Pre-allocate final tensors
        final_images = torch.zeros((total_samples,) + sample_shape, dtype=torch.float32)
        final_labels = torch.zeros((total_samples,) + sample_shape, dtype=torch.float32)
        final_masks = torch.zeros((total_samples,) + sample_shape, dtype=torch.float32)
        
        # Fill tensors incrementally
        current_idx = 0
        for temp_file in tqdm(temp_files, desc="Loading temp files"):
            if os.path.exists(temp_file):
                temp_data = torch.load(temp_file)
                batch_size = temp_data['images'].shape[0]
                
                # Copy data to pre-allocated tensors
                final_images[current_idx:current_idx + batch_size] = temp_data['images']
                final_labels[current_idx:current_idx + batch_size] = temp_data['labels']
                final_masks[current_idx:current_idx + batch_size] = temp_data['masks']
                
                current_idx += batch_size
                
                # Clear temp data
                del temp_data
                gc.collect()
        
        print("Creating final dataset...")
        final_data = {
            'images': final_images,
            'labels': final_labels,
            'masks': final_masks
        }
        
        # Save final data
        print(f"Saving final dataset to {final_path}...")
        torch.save(final_data, final_path)
        print(f"Saved final dataset: {final_data['images'].shape[0]} samples")
        
        # Clean up temp files
        print("Cleaning up temporary files...")
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return final_data
    
    # Create output paths
    os.makedirs('./data', exist_ok=True)
    train_path = f'./data/train_data_mixed.pt'
    val_path = f'./data/val_data_mixed.pt'
    
    # Process training data
    print("\n" + "="*50)
    print("PROCESSING TRAINING DATA")
    print("="*50)
    
    train_temp_files = []
    
    # Process cirrhotic training data (use all available)
    if cirrhotic_train_available > 0:
        print(f"\nProcessing cirrhotic training data...")
        temp_path = train_path.replace('.pt', '_cirrhotic_temp.pt')
        cirrhotic_temp_files = process_and_save_data(cirrhotic_groups['train'], "cirrhotic", target_size, temp_path, batch_size=2)
        train_temp_files.extend(cirrhotic_temp_files)
    
    # Process healthy training data (use all available)
    if healthy_train_available > 0:
        print(f"\nProcessing healthy training data...")
        temp_path = train_path.replace('.pt', '_healthy_temp.pt')
        healthy_temp_files = process_and_save_data(healthy_groups['train'], "healthy", target_size, temp_path, batch_size=2)
        train_temp_files.extend(healthy_temp_files)
    
    # Combine training data
    train_data = combine_temp_files(train_temp_files, train_path)
    
    # Process validation data
    print("\n" + "="*50)
    print("PROCESSING VALIDATION DATA")
    print("="*50)
    
    val_temp_files = []
    
    # Process cirrhotic validation data (use all available)
    if cirrhotic_valid_available > 0:
        print(f"\nProcessing cirrhotic validation data...")
        temp_path = val_path.replace('.pt', '_cirrhotic_temp.pt')
        cirrhotic_temp_files = process_and_save_data(cirrhotic_groups['valid'], "cirrhotic", target_size, temp_path, batch_size=2)
        val_temp_files.extend(cirrhotic_temp_files)
    
    # Process healthy validation data (use all available)
    if healthy_valid_available > 0:
        print(f"\nProcessing healthy validation data...")
        temp_path = val_path.replace('.pt', '_healthy_temp.pt')
        healthy_temp_files = process_and_save_data(healthy_groups['valid'], "healthy", target_size, temp_path, batch_size=2)
        val_temp_files.extend(healthy_temp_files)
    
    # Combine validation data
    val_data = combine_temp_files(val_temp_files, val_path)
    
    # Process test data
    print("\n" + "="*50)
    print("PROCESSING TEST DATA")
    print("="*50)
    
    test_temp_files = []
    
    # Process cirrhotic test data (use all available)
    if cirrhotic_test_available > 0:
        print(f"\nProcessing cirrhotic test data...")
        test_path = train_path.replace('train', 'test')
        temp_path = test_path.replace('.pt', '_cirrhotic_temp.pt')
        cirrhotic_temp_files = process_and_save_data(cirrhotic_groups['test'], "cirrhotic", target_size, temp_path, batch_size=2)
        test_temp_files.extend(cirrhotic_temp_files)
    
    # Process healthy test data (use all available)
    if healthy_test_available > 0:
        print(f"\nProcessing healthy test data...")
        test_path = train_path.replace('train', 'test')
        temp_path = test_path.replace('.pt', '_healthy_temp.pt')
        healthy_temp_files = process_and_save_data(healthy_groups['test'], "healthy", target_size, temp_path, batch_size=2)
        test_temp_files.extend(healthy_temp_files)
    
    # Combine test data
    test_path = train_path.replace('train', 'test')
    test_data = combine_temp_files(test_temp_files, test_path)
    
    print(f"Mixed dataset created successfully!")
    print(f"Training samples: {train_data['images'].shape[0]}")
    print(f"Validation samples: {val_data['images'].shape[0]}")
    print(f"Test samples: {test_data['images'].shape[0]}")
    print(f"Volume shape: {train_data['images'].shape[1:]}")
    
    # Print some statistics
    print("\nData statistics:")
    print(f"Image range: [{train_data['images'].min():.3f}, {train_data['images'].max():.3f}]")
    print(f"GT mask range: [{train_data['labels'].min():.3f}, {train_data['labels'].max():.3f}]")
    print(f"Pred mask range: [{train_data['masks'].min():.3f}, {train_data['masks'].max():.3f}]")
    
    print(f"\nFinal distribution:")
    print(f"Train: {cirrhotic_train_available} cirrhotic + {healthy_train_available} healthy = {train_data['images'].shape[0]} total")
    print(f"Valid: {cirrhotic_valid_available} cirrhotic + {healthy_valid_available} healthy = {val_data['images'].shape[0]} total")
    print(f"Test: {cirrhotic_test_available} cirrhotic + {healthy_test_available} healthy = {test_data['images'].shape[0]} total")
    
    print("\nData preparation completed successfully!")
    print("You can now run train_segment.py with:")
    print(f"python train_segment.py --train_data_path {train_path} --val_data_path {val_path} --simulate_slice_annot --no_wandb conv_net --num_epochs 100 --batch_size 4")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create mixed healthy/cirrhotic dataset')
    parser.add_argument('--target_size', type=int, nargs=3, default=[192, 192, 192],
                       help='Target volume size (H W D)')
    
    args = parser.parse_args()
    
    create_full_dataset(target_size=tuple(args.target_size))
