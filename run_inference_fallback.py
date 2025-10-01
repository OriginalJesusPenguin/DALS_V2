#!/usr/bin/env python3
"""
Run inference using fallback metrics module to avoid C++ compilation issues.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Replace the metrics module with our fallback version
import util.metrics_no_cpp as metrics_module
sys.modules['util.metrics'] = metrics_module

# Now import and run the inference
if __name__ == '__main__':
    print("Running inference with fallback metrics (no C++ compilation)...")
    
    try:
        from inference_meshdecoder import main
        
        # Set up minimal arguments for testing
        sys.argv = [
            'run_inference_fallback.py',
            '--data_path', '/home/ralbe/pyhppc_project/cirr_segm_clean/unit_sphere_meshes',
            '--checkpoint_dir', '.',
            '--latent_mode', 'global',
            '--max_iters', '5',
            '--num_point_samples', '1000'
        ]
        
        print("Starting inference...")
        main()
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
