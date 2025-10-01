#!/usr/bin/env python3
"""
Simple test script to run inference without C++ extension issues.
This bypasses the triangle self-intersection calculations.
"""

import sys
import os
import torch
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Mock the C++ extension to avoid compilation issues
class MockCTri:
    @staticmethod
    def triangle_self_intersections(verts, faces):
        # Return zeros instead of computing actual intersections
        return torch.zeros(faces.shape[0], dtype=torch.int32)

# Mock the metrics module
sys.modules['util.metrics'] = type('MockMetrics', (), {
    'point_metrics': lambda *args, **kwargs: {},
    'self_intersections': lambda meshes: (torch.zeros(len(meshes)), torch.zeros(1))
})()

# Mock the C++ extension
sys.modules['_C_tri'] = MockCTri()

# Now import and run the inference
if __name__ == '__main__':
    print("Running inference with mocked C++ extension...")
    
    # Import after mocking
    from inference_meshdecoder import main
    
    # Run with minimal arguments
    sys.argv = [
        'test_inference_simple.py',
        '--data_path', '/home/ralbe/pyhppc_project/cirr_segm_clean/unit_sphere_meshes',
        '--checkpoint_dir', '.',
        '--latent_mode', 'global',
        '--max_iters', '10'  # Reduced for testing
    ]
    
    try:
        main()
        print("Inference completed successfully!")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
