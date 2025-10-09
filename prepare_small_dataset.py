#!/usr/bin/env python3
"""
Create a small dataset for testing purposes.
"""

import os
import sys
import torch
import numpy as np
from prepare_cirrhotic_data import get_matching_files, process_batch

def create_small_dataset():
    """Create a small dataset with just 10 samples for testing."""
    
    # Define paths
    images_dir = '/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/processed/cirrhotic/T1_images'
    gt_dir = '/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/processed/cirrhotic/T1_masks/GT'
    pred_dir = '/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/processed/cirrhotic/T1_masks/AttentionUNet_ROI'
    
    print("Creating small dataset...")
    
    # Get matching files
    matching_files = get_matching_files(images_dir, gt_dir, pred_dir)
    print(f"Found {len(matching_files)} matching files")
    
    # Take first 10 files for testing
    test_files = matching_files[:10]
    print(f"Using {len(test_files)} files for testing")
    
    # Process the files
    batch_data = process_batch(test_files, target_size=(64, 64, 64))
    
    print(f"Processed {len(batch_data['images'])} samples")
    
    if len(batch_data['images']) > 0:
        # Split into train/val (8 train, 2 val) and stack tensors for simulate_slice_annot
        train_data = {
            'images': torch.stack(batch_data['images'][:8]),
            'labels': torch.stack(batch_data['labels'][:8]),
            'masks': torch.stack(batch_data['masks'][:8])
        }
        
        val_data = {
            'images': torch.stack(batch_data['images'][8:]),
            'labels': torch.stack(batch_data['labels'][8:]),
            'masks': torch.stack(batch_data['masks'][8:])
        }
        
        # Save data
        os.makedirs('./data', exist_ok=True)
        
        train_path = './data/train_data.pt'
        val_path = './data/val_data.pt'
        
        print(f"Saving training data to {train_path}...")
        torch.save(train_data, train_path)
        
        print(f"Saving validation data to {val_path}...")
        torch.save(val_data, val_path)
        
        print(f"Small dataset created successfully!")
        print(f"Training samples: {train_data['images'].shape[0]}")
        print(f"Validation samples: {val_data['images'].shape[0]}")
        print(f"Volume shape: {train_data['images'].shape[1:]}")
        
        # Test loading
        print("\nTesting data loading...")
        loaded_train = torch.load(train_path)
        loaded_val = torch.load(val_path)
        
        print(f"Loaded train data type: {type(loaded_train)}")
        print(f"Loaded val data type: {type(loaded_val)}")
        print(f"Train data keys: {list(loaded_train.keys())}")
        print(f"Train data shapes: {[loaded_train['images'].shape, loaded_train['labels'].shape, loaded_train['masks'].shape]}")
        
        print("\nData preparation completed successfully!")
        print("You can now run train_segment.py with:")
        print("python train_segment.py conv_net --train_data_path ./data/train_data.pt --val_data_path ./data/val_data.pt --simulate_slice_annot")
        
    else:
        print("No data processed!")

if __name__ == '__main__':
    create_small_dataset()
