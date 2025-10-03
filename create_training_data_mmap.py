import torch
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import numpy as np
import os
from scipy.ndimage import zoom

def create_training_data_mmap(df, output_file, desc="Processing", batch_size=5, target_size=(64, 64, 64)):
    """Create training data using memory mapping to avoid OOM errors."""
    # filter out rows where T1_img, T1_mask, T1_mask_UNet are None
    df = df[df['T1_img'].notna() & df['T1_mask'].notna() & df['T1_mask_UNet'].notna()]
    print(f"Filtered {len(df)} rows from {desc} dataset")
    
    # Get dimensions from first sample and calculate downsampling factor
    first_img = nib.load(df.iloc[0]['T1_img']).get_fdata()
    original_shape = first_img.shape
    print(f"Original image shape: {original_shape}")
    
    # Calculate zoom factors for downsampling
    zoom_factors = tuple(target_size[i] / original_shape[i] for i in range(len(original_shape)))
    print(f"Downsampling from {original_shape} to {target_size}")
    print(f"Zoom factors: {zoom_factors}")
    
    # Create memory-mapped arrays with target size
    n_samples = len(df)
    mmap_images = np.memmap(f'{output_file}_images.dat', dtype='float32', mode='w+', 
                           shape=(n_samples, *target_size))
    mmap_labels = np.memmap(f'{output_file}_labels.dat', dtype='int64', mode='w+', 
                           shape=(n_samples, *target_size))
    mmap_masks = np.memmap(f'{output_file}_masks.dat', dtype='int64', mode='w+', 
                          shape=(n_samples, *target_size))
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc=desc):
        batch_df = df.iloc[i:i+batch_size]
        
        for j, (_, row) in enumerate(batch_df.iterrows()):
            idx = i + j
            
            # Load data
            img = nib.load(row['T1_img']).get_fdata().astype('float32')
            mask = nib.load(row['T1_mask']).get_fdata().astype('int64')
            unet_mask = nib.load(row['T1_mask_UNet']).get_fdata().astype('int64')
            
            # Downsample images using linear interpolation
            img_downsampled = zoom(img, zoom_factors, order=1, mode='nearest')
            
            # Downsample masks using nearest neighbor interpolation (preserve labels)
            mask_downsampled = zoom(mask, zoom_factors, order=0, mode='nearest')
            unet_mask_downsampled = zoom(unet_mask, zoom_factors, order=0, mode='nearest')
            
            # Store downsampled data
            mmap_images[idx] = img_downsampled
            mmap_labels[idx] = mask_downsampled
            mmap_masks[idx] = unet_mask_downsampled
    
    # Flush to disk
    mmap_images.flush()
    mmap_labels.flush()
    mmap_masks.flush()
    
    # Convert memory-mapped arrays to regular tensors and save in expected format
    print("Converting memory-mapped data to tensors...")
    final_data = {
        'images': torch.from_numpy(np.array(mmap_images)),
        'labels': torch.from_numpy(np.array(mmap_labels)),
        'masks': torch.from_numpy(np.array(mmap_masks))
    }
    
    # Save in the format expected by train_segment.py
    torch.save(final_data, f'{output_file}.pt')
    
    # Clean up memory-mapped files
    os.unlink(f'{output_file}_images.dat')
    os.unlink(f'{output_file}_labels.dat')
    os.unlink(f'{output_file}_masks.dat')
    
    print(f"Saved {n_samples} samples to {output_file}.pt")

def load_mmap_data(metadata_file):
    """Load data from memory-mapped files."""
    metadata = torch.load(metadata_file)
    
    images = np.memmap(metadata['images_file'], dtype='float32', mode='r', 
                      shape=(metadata['n_samples'], *metadata['img_shape']))
    labels = np.memmap(metadata['labels_file'], dtype='int64', mode='r', 
                      shape=(metadata['n_samples'], *metadata['img_shape']))
    masks = np.memmap(metadata['masks_file'], dtype='int64', mode='r', 
                     shape=(metadata['n_samples'], *metadata['img_shape']))
    
    return {
        'images': torch.from_numpy(images),
        'labels': torch.from_numpy(labels),
        'masks': torch.from_numpy(masks)
    }

def main():
    healthy_csv = '/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/healthy_dataset.csv'
    cirrhotic_csv = '/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/cirrhotic_dataset.csv'
    cirrhotic_output_file = '/home/ralbe/DALS/mesh_autodecoder/training_data_for_segm_cirrhotic.pt'
    healthy_output_file = '/home/ralbe/DALS/mesh_autodecoder/training_data_for_segm_healthy.pt'

    print("Loading CSV files...")
    healthy_df = pd.read_csv(healthy_csv)
    cirrhotic_df = pd.read_csv(cirrhotic_csv)

    print("Creating cirrhotic training data...")
    create_training_data_mmap(cirrhotic_df, cirrhotic_output_file, desc="Cirrhotic", batch_size=5, target_size=(64, 64, 64))

    print("Creating healthy training data...")
    create_training_data_mmap(healthy_df, healthy_output_file, desc="Healthy", batch_size=5, target_size=(64, 64, 64))

if __name__ == "__main__":
    main()
