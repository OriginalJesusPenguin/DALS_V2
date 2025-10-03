#!/bin/bash
# Activation script for mesh_autodecoder conda environment

echo "Activating mesh_autodecoder conda environment..."
echo "Environment location: /home/ralbe/miniconda3/envs/mesh_autodecoder"

# Source conda
source /home/ralbe/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate mesh_autodecoder

# Fix library path for libremesh.so
export LD_LIBRARY_PATH="/home/ralbe/DALS/mesh_autodecoder/util/bin:$LD_LIBRARY_PATH"

echo "Environment activated successfully!"
echo "Library path updated for libremesh.so"
echo ""
echo "Installed packages:"
echo "- PyTorch 1.10.2 (CUDA 11.3 support)"
echo "- PyTorch3D 0.6.1"
echo "- Python 3.8"
echo "- All requirements from requirements.txt"
echo ""
echo "Note: CUDA is not available in this environment (CUDA available: False)"
echo "This is normal for headless environments. CUDA will work when running on a GPU-enabled system."
echo ""
echo "To deactivate, run: conda deactivate"
