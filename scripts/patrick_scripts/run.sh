#!/bin/bash
# Simple script to run training with proper environment setup

# Set library path for libremesh.so
export LD_LIBRARY_PATH="/home/ralbe/DALS/mesh_autodecoder/util/bin:$LD_LIBRARY_PATH"

# Run with the conda environment Python
exec /home/ralbe/miniconda3/envs/mesh_autodecoder/bin/python "$@"
