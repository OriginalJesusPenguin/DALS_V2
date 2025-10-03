#!/bin/bash
# Wrapper script to run training with proper library path

# Set the library path for libremesh.so
export LD_LIBRARY_PATH="/home/ralbe/DALS/mesh_autodecoder/util/bin:$LD_LIBRARY_PATH"

# Use the conda environment Python directly
exec /home/ralbe/miniconda3/envs/mesh_autodecoder/bin/python "$@"
