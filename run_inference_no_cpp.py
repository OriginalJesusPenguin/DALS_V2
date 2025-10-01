#!/usr/bin/env python3
"""
Wrapper script to run inference without C++ extension issues.
This script sets up the environment to avoid C++ compilation problems.
"""

import os
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set environment variables to avoid C++ compilation issues
os.environ['TORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock problematic C++ extensions before any imports
class MockCTri:
    @staticmethod
    def triangle_self_intersections(verts, faces):
        return torch.zeros(faces.shape[0], dtype=torch.int32)

# Mock the extension before importing
import torch
sys.modules['_C_tri'] = MockCTri()

# Now we can safely import and run
if __name__ == '__main__':
    print("Setting up inference environment...")
    
    # Import the inference script
    try:
        from inference_meshdecoder import main
        
        # Set up command line arguments
        sys.argv = [
            'run_inference_no_cpp.py',
            '--data_path', '/home/ralbe/pyhppc_project/cirr_segm_clean/unit_sphere_meshes',
            '--checkpoint_dir', '.',
            '--latent_mode', 'global',
            '--max_iters', '10',
            '--num_point_samples', '1000'
        ]
        
        print("Running inference...")
        main()
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
