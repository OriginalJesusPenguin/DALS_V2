# Mesh Autodecoder Setup Fix

## Problem
The original error was:
```
ImportError: libremesh.so: cannot open shared object file: No such file or directory
```

## Root Cause
The `libremesh.so` shared library exists in `/home/ralbe/DALS/mesh_autodecoder/util/bin/` but the system cannot find it because it's not in the library search path (`LD_LIBRARY_PATH`).

## Solution

### Option 1: Use the wrapper script (Recommended)
```bash
# Run any Python command with the wrapper script
./run_training.sh train.py mesh_decoder --data_path='your_data_path' [other_args]
```

### Option 2: Set library path manually
```bash
# Set the library path and run Python directly
export LD_LIBRARY_PATH="/home/ralbe/DALS/mesh_autodecoder/util/bin:$LD_LIBRARY_PATH"
/home/ralbe/miniconda3/envs/mesh_autodecoder/bin/python train.py mesh_decoder --data_path='your_data_path' [other_args]
```

### Option 3: Use the updated activation script
```bash
# The activation script now includes the library path fix
source activate_env.sh
python train.py mesh_decoder --data_path='your_data_path' [other_args]
```

## Files Created/Modified

1. **`run_training.sh`** - Wrapper script that automatically sets the library path
2. **`activate_env.sh`** - Updated to include library path fix
3. **`SETUP_FIX.md`** - This documentation

## Verification
The fix has been tested and the training script now runs without the `libremesh.so` error. You can verify by running:

```bash
./run_training.sh train.py mesh_decoder --help
```

## Notes
- The conda environment `mesh_autodecoder` is properly set up with all required packages
- PyTorch 1.10.2 and PyTorch3D 0.6.1 are installed and working
- All dependencies from `requirements.txt` are installed
- The library path issue only affects the `libremesh.so` shared library used for mesh remeshing operations
