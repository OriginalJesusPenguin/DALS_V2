# Mesh Preprocessing Pipeline

This document describes the comprehensive mesh preprocessing pipeline that creates 8 different preprocessing combinations for mesh decoder training.

## Overview

The pipeline creates 8 different preprocessing combinations by varying three parameters:

1. **Data Split**: `separate` vs `mixed`
2. **Coordinate Scaling**: `individual` vs `global`  
3. **Augmentation**: `noaug` vs `aug`

## Preprocessing Combinations

| Combination | Data Split | Scaling | Augmentation | Description |
|-------------|------------|---------|--------------|-------------|
| `separate_individual_noaug` | Separate | Individual | No | Healthy→train, Cirrhotic→val/test, Individual scaling, No augmentation |
| `separate_individual_aug` | Separate | Individual | Yes | Healthy→train, Cirrhotic→val/test, Individual scaling, 99x augmentation |
| `separate_global_noaug` | Separate | Global | No | Healthy→train, Cirrhotic→val/test, Global scaling, No augmentation |
| `separate_global_aug` | Separate | Global | Yes | Healthy→train, Cirrhotic→val/test, Global scaling, 99x augmentation |
| `mixed_individual_noaug` | Mixed | Individual | No | Stratified 70/15/15 split, Individual scaling, No augmentation |
| `mixed_individual_aug` | Mixed | Individual | Yes | Stratified 70/15/15 split, Individual scaling, 99x augmentation |
| `mixed_global_noaug` | Mixed | Global | No | Stratified 70/15/15 split, Global scaling, No augmentation |
| `mixed_global_aug` | Mixed | Global | Yes | Stratified 70/15/15 split, Global scaling, 99x augmentation |

## Data Split Methods

### Separate Split
- **Training**: All healthy meshes (55 meshes)
- **Validation**: Cirrhotic meshes not in test set (~127 meshes)
- **Test**: Cirrhotic meshes with specific test indices (128 meshes)

### Mixed Split
- **Training**: 70% of all meshes (stratified by disease severity)
- **Validation**: 15% of all meshes (stratified by disease severity)
- **Test**: 15% of all meshes (stratified by disease severity)
- **Stratification**: Based on 'Radiological Evaluation' column (values: 1, 2, 3)
- **Healthy patients**: Assigned severity = 0

## Coordinate Scaling Methods

### Individual Scaling
- Centers each mesh at its center of mass (COM)
- Scales each mesh to have maximum radius = 1.0
- Each mesh becomes a unit sphere
- Preserves individual shape characteristics

### Global Scaling
- Centers each mesh at its center of mass (COM)
- Finds global maximum radius across ALL meshes
- Scales all meshes by the same global maximum radius
- Preserves relative sizes between meshes

## Augmentation

### No Augmentation (`noaug`)
- Uses original meshes only
- No data augmentation applied

### With Augmentation (`aug`)
- Creates 99 augmented variants per original training mesh
- Uses PointWOLF augmentation with parameters:
  - `num_anchor=4`
  - `sample_type='fps'`
  - `sigma=0.5`
  - `R_range=10`
  - `S_range=2`
  - `T_range=0.25`
- Results in 100 total meshes per original (1 original + 99 augmented)

## Folder Structure

```
/home/ralbe/DALS/mesh_autodecoder/data/
├── separate_individual_noaug/
│   ├── train_meshes/
│   ├── val_meshes/
│   └── test_meshes/
├── separate_individual_aug/
│   ├── train_meshes/  (100x augmented)
│   ├── val_meshes/
│   └── test_meshes/
├── separate_global_noaug/
│   ├── train_meshes/
│   ├── val_meshes/
│   └── test_meshes/
├── separate_global_aug/
│   ├── train_meshes/  (100x augmented)
│   ├── val_meshes/
│   └── test_meshes/
├── mixed_individual_noaug/
│   ├── train_meshes/
│   ├── val_meshes/
│   └── test_meshes/
├── mixed_individual_aug/
│   ├── train_meshes/  (100x augmented)
│   ├── val_meshes/
│   └── test_meshes/
├── mixed_global_noaug/
│   ├── train_meshes/
│   ├── val_meshes/
│   └── test_meshes/
└── mixed_global_aug/
    ├── train_meshes/  (100x augmented)
    ├── val_meshes/
    └── test_meshes/
```

## Usage

### Run All Preprocessing Combinations

```bash
# Make the script executable
chmod +x /home/ralbe/DALS/mesh_autodecoder/run_preprocessing.sh

# Run all preprocessing combinations
/home/ralbe/DALS/mesh_autodecoder/run_preprocessing.sh
```

### Run Individual Combination

```python
from preprocess_meshes import MeshPreprocessor

preprocessor = MeshPreprocessor()
preprocessor.process_combination("mixed", "individual", "aug")
```

## Input Data

- **Healthy Meshes**: `/home/ralbe/pyhppc_project/cirr_segm_clean/healthy_T1_meshes/`
- **Cirrhotic Meshes**: `/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_T1_meshes/`
- **Disease Severity**: `/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/cirrhotic_dataset.csv`

## Output

Each preprocessing combination creates:
- **Train meshes**: Scaled and optionally augmented
- **Val meshes**: Scaled only
- **Test meshes**: Scaled only

The pipeline handles:
- ✅ Data loading and validation
- ✅ Patient ID extraction from filenames
- ✅ Disease severity mapping from CSV
- ✅ Stratified splitting for mixed approach
- ✅ Individual and global coordinate scaling
- ✅ PointWOLF augmentation for training data
- ✅ Organized folder structure
- ✅ Error handling and progress tracking

## Expected Results

- **Total meshes**: 365 (55 healthy + 310 cirrhotic)
- **Augmented training sets**: ~36,500 meshes (365 × 100)
- **Processing time**: ~2-4 hours depending on augmentation
- **Disk space**: ~50-100 GB for all combinations
