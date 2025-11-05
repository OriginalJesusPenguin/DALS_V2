#!/bin/bash

# Diagnostic script to check CUDA environment in batch jobs

echo "=== CUDA Environment Diagnostics ==="
echo ""

echo "1. SLURM GPU Allocation:"
echo "   SLURM_GPU_ALLOC: ${SLURM_GPU_ALLOC:-not set}"
echo "   SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-not set}"
echo ""

echo "2. CUDA_VISIBLE_DEVICES:"
echo "   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

echo "3. Python/Conda Environment:"
which python
python --version
echo ""

echo "4. PyTorch Import Test:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'Device count: {torch.cuda.device_count()}'); import os; print(f'CUDA_VISIBLE_DEVICES in Python: {os.environ.get(\"CUDA_VISIBLE_DEVICES\", \"not set\")}')"
echo ""

echo "5. nvidia-smi (if available):"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader 2>/dev/null || echo "   nvidia-smi not available or failed"
echo ""

echo "6. CUDA Library Paths:"
echo "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo ""

echo "7. Module system (if used):"
module list 2>/dev/null || echo "   No module system"
echo ""

echo "=== End Diagnostics ==="

