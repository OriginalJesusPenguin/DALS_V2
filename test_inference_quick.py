#!/usr/bin/env python3
"""
Quick test script to verify inference works without C++ extension issues.
"""

import os
import sys
import warnings

# Suppress all warnings to avoid clutter
warnings.filterwarnings("ignore")

# Set environment variables
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that we can import the necessary modules."""
    try:
        print("Testing imports...")
        import torch
        print("✓ PyTorch imported")
        
        # Test metrics import
        from util.metrics import point_metrics, self_intersections
        print("✓ Metrics module imported")
        
        # Test inference import
        from inference_meshdecoder import main
        print("✓ Inference module imported")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_inference():
    """Test running inference with minimal parameters."""
    try:
        print("\nTesting inference...")
        
        # Set up arguments
        sys.argv = [
            'test_inference_quick.py',
            '--data_path', '/home/ralbe/pyhppc_project/cirr_segm_clean/unit_sphere_meshes',
            '--checkpoint_dir', '.',
            '--latent_mode', 'global',
            '--max_iters', '3',  # Very short for testing
            '--num_point_samples', '100'
        ]
        
        from inference_meshdecoder import main
        main()
        print("✓ Inference completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Inference error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Testing inference setup...")
    
    if test_imports():
        print("\nAll imports successful!")
        if test_inference():
            print("\n🎉 All tests passed! Inference is working correctly.")
        else:
            print("\n❌ Inference test failed.")
    else:
        print("\n❌ Import test failed.")
