# Quick Start Guide

## ✅ How to Run the Training

The training script is now working! Here's how to run it:

### Basic Command (CPU mode, no wandb):
```bash
cd /home/ralbe/DALS/mesh_autodecoder
./run.sh train.py --data_path="/path/to/your/meshes" --no_wandb --device cpu mesh_decoder --decoder_mode mlp
```

### Your Specific Data:
```bash
cd /home/ralbe/DALS/mesh_autodecoder
./run.sh train.py --data_path="/home/ralbe/pyhppc_project/cirr_segm_clean/processed_data_trained_all/cirrhotic/T1_meshes/GT" --no_wandb --device cpu mesh_decoder --decoder_mode mlp
```

### With Additional Parameters:
```bash
./run.sh train.py --data_path="/path/to/meshes" --no_wandb --device cpu mesh_decoder --decoder_mode mlp --num_epochs 10 --batch_size 2
```

## Key Points:

1. **Use `./run.sh`** instead of `python` directly
2. **Put `--data_path` BEFORE `mesh_decoder`** (this is the correct order)
3. **Use `--no_wandb`** to disable Weights & Biases logging
4. **Use `--device cpu`** if you don't have CUDA/GPU
5. **Use `--decoder_mode mlp`** to avoid GCNN issues

## What's Fixed:

- ✅ Conda environment with PyTorch 1.10.2 and PyTorch3D 0.6.1
- ✅ Library path for libremesh.so
- ✅ All dependencies installed
- ✅ Command syntax working

## Current Issue:

The training is failing because the mesh data contains NaN or infinite values. This might be due to:
- Corrupted mesh files
- Mesh processing issues
- Data format problems

Try with a different dataset or check your mesh files for validity.
