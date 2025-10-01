#!/usr/bin/env python3
"""
Test to check PyTorch3D installation and CUDA compilation status.
This test works even if CUDA drivers aren't available.
"""

import torch
import sys

def test_pytorch3d_installation():
    print("Testing PyTorch3D installation and CUDA compilation...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch compiled with CUDA: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    # Check if PyTorch has CUDA support compiled in
    has_cuda_support = torch.version.cuda is not None
    print(f"PyTorch has CUDA support compiled: {has_cuda_support}")
    
    if not has_cuda_support:
        print("❌ PyTorch was not compiled with CUDA support")
        return False
    
    # Test basic PyTorch3D import
    try:
        import pytorch3d
        print(f"PyTorch3D version: {pytorch3d.__version__}")
        print("✅ PyTorch3D imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyTorch3D: {e}")
        return False
    
    # Test PyTorch3D operations on CPU
    try:
        from pytorch3d.structures import Meshes
        from pytorch3d.ops import sample_points_from_meshes
        from pytorch3d.ops.graph_conv import gather_scatter
        
        print("\nTesting PyTorch3D operations on CPU...")
        
        # Create a simple mesh
        verts = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=torch.float32)
        faces = torch.tensor([[[0, 1, 2]]], dtype=torch.long)
        mesh = Meshes(verts=verts, faces=faces)
        
        # Test sampling
        result = sample_points_from_meshes(mesh, num_samples=100, return_normals=True)
        if isinstance(result, tuple) and len(result) == 2:
            print("✅ PyTorch3D sampling works (with normals)")
        else:
            print("✅ PyTorch3D sampling works (points only)")
        
        # Test graph operations
        edges = mesh.edges_packed()
        test_input = torch.randn(edges.shape[0], 3)
        output = gather_scatter(test_input, edges, False)
        print("✅ PyTorch3D graph operations work")
        
    except Exception as e:
        print(f"❌ PyTorch3D operations failed: {e}")
        return False
    
    # Check if PyTorch3D was compiled with CUDA support
    print("\nChecking PyTorch3D CUDA compilation...")
    try:
        # Try to create a tensor on CUDA (this will fail if no driver, but that's OK)
        test_tensor = torch.randn(2, 3)
        if torch.cuda.is_available():
            print("✅ CUDA is available in PyTorch")
            # Test if PyTorch3D can work with CUDA tensors
            try:
                mesh_gpu = mesh.to('cuda')
                edges_gpu = mesh_gpu.edges_packed()
                test_input_gpu = torch.randn(edges_gpu.shape[0], 3, device='cuda')
                output_gpu = gather_scatter(test_input_gpu, edges_gpu, False)
                print("✅ PyTorch3D CUDA operations work")
                return True
            except Exception as e:
                print(f"❌ PyTorch3D CUDA operations failed: {e}")
                print("This suggests PyTorch3D was not compiled with CUDA support")
                return False
        else:
            print("⚠️  CUDA is not available (no driver or wrong environment)")
            print("PyTorch3D appears to be installed correctly, but CUDA testing is not possible")
            return True  # Installation is OK, just can't test CUDA
    
    except Exception as e:
        print(f"❌ Error checking CUDA support: {e}")
        return False

if __name__ == "__main__":
    success = test_pytorch3d_installation()
    if success:
        print("\n🎉 PyTorch3D installation appears to be working!")
    else:
        print("\n❌ PyTorch3D installation has issues")
    sys.exit(0 if success else 1)
