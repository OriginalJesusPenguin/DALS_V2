#!/usr/bin/env python3
"""
Data preparation script for cirrhotic liver segmentation data.
Converts NIfTI files to PyTorch format required by train_segment.py.

This script loads:
- Images from T1_images directory
- Ground truth masks from T1_masks/GT directory  
- Model predictions from T1_masks/AttentionUNet_ROI directory (used as slice annotations)

And saves them as .pt files for training.
"""

import os
import sys
import argparse
import numpy as np
import torch
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split


def load_nifti_volume(filepath, target_size=(192, 192, 192)):
    """
    Load a NIfTI volume and resize to target size.
    
    Args:
        filepath: Path to NIfTI file
        target_size: Target size (H, W, D)
    
    Returns:
        numpy array of shape target_size
    """
    try:
        # Load NIfTI file
        nii = nib.load(filepath)
        volume = nii.get_fdata()
        
        # Ensure we have the right data type
        if volume.dtype != np.float32:
            volume = volume.astype(np.float32)
        
        # Resize if necessary (simple nearest neighbor for masks, linear for images)
        if volume.shape != target_size:
            from scipy.ndimage import zoom
            zoom_factors = [target_size[i] / volume.shape[i] for i in range(3)]
            volume = zoom(volume, zoom_factors, order=0 if 'mask' in str(filepath).lower() else 1)
        
        return volume
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def get_matching_files(images_dir, gt_dir, pred_dir):
    """
    Get matching files from all three directories.
    
    Returns:
        List of tuples (image_path, gt_path, pred_path, sample_id)
    """
    image_files = set(os.listdir(images_dir))
    gt_files = set(os.listdir(gt_dir))
    pred_files = set(os.listdir(pred_dir))
    
    # Find common files
    common_files = image_files.intersection(gt_files).intersection(pred_files)
    
    matching_files = []
    for filename in sorted(common_files):
        if filename.endswith('.nii.gz'):
            image_path = os.path.join(images_dir, filename)
            gt_path = os.path.join(gt_dir, filename)
            pred_path = os.path.join(pred_dir, filename)
            sample_id = filename.replace('.nii.gz', '')
            
            matching_files.append((image_path, gt_path, pred_path, sample_id))
    
    return matching_files


def process_batch(files_batch, target_size=(192, 192, 192)):
    """
    Process a batch of files to avoid memory issues.
    
    Args:
        files_batch: List of (image_path, gt_path, pred_path, sample_id) tuples
        target_size: Target volume size
    
    Returns:
        Dictionary with 'images', 'labels', 'masks' keys containing tensors
    """
    batch_data = {
        'images': [],
        'labels': [],
        'masks': []
    }
    
    for image_path, gt_path, pred_path, sample_id in files_batch:
        # Load image
        image = load_nifti_volume(image_path, target_size)
        if image is None:
            continue
            
        # Load ground truth mask
        gt_mask = load_nifti_volume(gt_path, target_size)
        if gt_mask is None:
            continue
            
        # Load prediction mask (used as slice annotation)
        pred_mask = load_nifti_volume(pred_path, target_size)
        if pred_mask is None:
            continue
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()
        gt_tensor = torch.from_numpy(gt_mask).float()
        pred_tensor = torch.from_numpy(pred_mask).float()
        
        # Normalize image to [0, 1] range
        image_min = image_tensor.min()
        image_max = image_tensor.max()
        if image_max > image_min:
            image_tensor = (image_tensor - image_min) / (image_max - image_min)
        
        # Ensure masks are binary (0 or 1)
        gt_tensor = (gt_tensor > 0.5).float()
        pred_tensor = (pred_tensor > 0.5).float()
        
        # Don't add channel dimension - let MONAI handle it with AddChanneld transform
        
        batch_data['images'].append(image_tensor)
        batch_data['labels'].append(gt_tensor)
        batch_data['masks'].append(pred_tensor)
        
        # Clean up memory
        del image, gt_mask, pred_mask, image_tensor, gt_tensor, pred_tensor
        gc.collect()
    
    return batch_data


def prepare_data(images_dir, gt_dir, pred_dir, output_dir, batch_size=10, 
                train_ratio=0.8, target_size=(192, 192, 192)):
    """
    Prepare training data from NIfTI files.
    
    Args:
        images_dir: Directory containing T1 images
        gt_dir: Directory containing ground truth masks
        pred_dir: Directory containing prediction masks
        output_dir: Directory to save .pt files
        batch_size: Batch size for processing (to manage memory)
        train_ratio: Ratio of data to use for training
        target_size: Target volume size
    """
    
    print("Finding matching files...")
    matching_files = get_matching_files(images_dir, gt_dir, pred_dir)
    print(f"Found {len(matching_files)} matching files")
    
    if len(matching_files) == 0:
        print("No matching files found!")
        return
    
    # Split into train/val
    train_files, val_files = train_test_split(
        matching_files, 
        train_size=train_ratio, 
        random_state=42
    )
    
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    
    # Process training data
    print("Processing training data...")
    train_images = []
    train_labels = []
    train_masks = []
    
    for i in tqdm(range(0, len(train_files), batch_size), desc="Processing train batches"):
        batch_files = train_files[i:i+batch_size]
        batch_data = process_batch(batch_files, target_size)
        
        train_images.extend(batch_data['images'])
        train_labels.extend(batch_data['labels'])
        train_masks.extend(batch_data['masks'])
        
        # Clear batch data from memory
        del batch_data
        gc.collect()
    
    # Create list of dicts format for training data
    train_data = []
    for i in range(len(train_images)):
        train_data.append({
            'images': train_images[i],
            'labels': train_labels[i],
            'masks': train_masks[i]
        })
    
    # Process validation data
    print("Processing validation data...")
    val_images = []
    val_labels = []
    val_masks = []
    
    for i in tqdm(range(0, len(val_files), batch_size), desc="Processing val batches"):
        batch_files = val_files[i:i+batch_size]
        batch_data = process_batch(batch_files, target_size)
        
        val_images.extend(batch_data['images'])
        val_labels.extend(batch_data['labels'])
        val_masks.extend(batch_data['masks'])
        
        # Clear batch data from memory
        del batch_data
        gc.collect()
    
    # Create list of dicts format for validation data
    val_data = []
    for i in range(len(val_images)):
        val_data.append({
            'images': val_images[i],
            'labels': val_labels[i],
            'masks': val_masks[i]
        })
    
    # Save data
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_data.pt')
    val_path = os.path.join(output_dir, 'val_data.pt')
    
    print(f"Saving training data to {train_path}...")
    torch.save(train_data, train_path)
    
    print(f"Saving validation data to {val_path}...")
    torch.save(val_data, val_path)
    
    print(f"Data preparation complete!")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Volume shape: {train_data[0]['images'].shape}")
    
    # Print some statistics
    print("\nData statistics:")
    all_images = torch.stack([item['images'] for item in train_data])
    all_labels = torch.stack([item['labels'] for item in train_data])
    all_masks = torch.stack([item['masks'] for item in train_data])
    print(f"Image range: [{all_images.min():.3f}, {all_images.max():.3f}]")
    print(f"GT mask range: [{all_labels.min():.3f}, {all_labels.max():.3f}]")
    print(f"Pred mask range: [{all_masks.min():.3f}, {all_masks.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description='Prepare cirrhotic liver data for training')
    parser.add_argument('--images_dir', type=str, 
                       default='/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/processed/cirrhotic/T1_images',
                       help='Directory containing T1 images')
    parser.add_argument('--gt_dir', type=str,
                       default='/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/processed/cirrhotic/T1_masks/GT',
                       help='Directory containing ground truth masks')
    parser.add_argument('--pred_dir', type=str,
                       default='/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/processed/cirrhotic/T1_masks/AttentionUNet_ROI',
                       help='Directory containing prediction masks')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for .pt files')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Batch size for processing (to manage memory)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--target_size', type=int, nargs=3, default=[192, 192, 192],
                       help='Target volume size (H W D)')
    
    args = parser.parse_args()
    
    # Validate directories
    for dir_path in [args.images_dir, args.gt_dir, args.pred_dir]:
        if not os.path.exists(dir_path):
            print(f"Error: Directory {dir_path} does not exist!")
            sys.exit(1)
    
    print("Data preparation configuration:")
    print(f"Images directory: {args.images_dir}")
    print(f"GT directory: {args.gt_dir}")
    print(f"Pred directory: {args.pred_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Target size: {args.target_size}")
    print()
    
    # Prepare data
    prepare_data(
        images_dir=args.images_dir,
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        target_size=tuple(args.target_size)
    )


if __name__ == '__main__':
    main()
