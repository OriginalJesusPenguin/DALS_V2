#!/usr/bin/env python3
"""
Simple test to check if PyTorch3D has CUDA support.
This test creates a basic mesh and tries to perform operations on GPU.
"""

import torch
import sys

def test_pytorch3d_cuda():
    print("Testing PyTorch3D CUDA support...")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check PyTorch CUDA compilation
    print(f"PyTorch compiled with CUDA: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    
    # Test basic PyTorch CUDA operations
    print("\nTesting PyTorch CUDA operations...")
    try:
        # Test tensor creation on GPU
        test_tensor = torch.randn(10, 10).cuda()
        result = torch.mm(test_tensor, test_tensor.t())
        print("✅ PyTorch CUDA tensor operations work")
        
        # Test CUDA availability
        if not torch.cuda.is_available():
            print("❌ CUDA is not available on this system")
            return False
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        
        # Test CUDA memory
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"❌ PyTorch CUDA operations failed: {e}")
        return False
    
    try:
        # Import PyTorch3D
        import pytorch3d
        print(f"PyTorch3D version: {pytorch3d.__version__}")
        
        # Test basic PyTorch3D operations on CPU first
        from pytorch3d.structures import Meshes
        from pytorch3d.ops import sample_points_from_meshes
        
        print("\nTesting basic PyTorch3D operations...")
        
        # Create a simple mesh on CPU
        verts = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=torch.float32)
        faces = torch.tensor([[[0, 1, 2]]], dtype=torch.long)
        mesh_cpu = Meshes(verts=verts, faces=faces)
        
        # Test sampling on CPU
        result_cpu = sample_points_from_meshes(mesh_cpu, num_samples=100, return_normals=True)
        if isinstance(result_cpu, tuple) and len(result_cpu) == 2:
            points_cpu, normals_cpu = result_cpu
            print("✅ CPU operations work (with normals)")
        else:
            points_cpu = result_cpu
            print("✅ CPU operations work (points only)")
        
        # Now test on GPU
        print("\nTesting PyTorch3D operations on GPU...")
        device = torch.device('cuda')
        
        # Move mesh to GPU
        mesh_gpu = mesh_cpu.to(device)
        print("✅ Mesh moved to GPU successfully")
        
        # Try sampling on GPU
        result_gpu = sample_points_from_meshes(mesh_gpu, num_samples=100, return_normals=True)
        if isinstance(result_gpu, tuple) and len(result_gpu) == 2:
            points_gpu, normals_gpu = result_gpu
            print("✅ GPU sampling operations work (with normals)")
        else:
            points_gpu = result_gpu
            print("✅ GPU sampling operations work (points only)")
        
        # Test graph convolution operations (the one that was failing)
        try:
            from pytorch3d.ops import gather_scatter
            from pytorch3d.structures import Meshes
            
            print("\nTesting PyTorch3D graph convolution operations...")
            
            # Create a simple mesh for graph operations
            verts = torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
            faces = torch.tensor([[[0, 1, 2], [0, 1, 3]]], dtype=torch.long)
            
            # Test on CPU first
            mesh_cpu = Meshes(verts=verts, faces=faces)
            edges_cpu = mesh_cpu.edges_packed()
            test_input_cpu = torch.randn(edges_cpu.shape[0], 3)
            output_cpu = gather_scatter(test_input_cpu, edges_cpu, directed=False, backward=False)
            print("✅ gather_scatter operation works on CPU")
            
            # Test on GPU
            mesh_gpu = mesh_cpu.to(device)
            edges_gpu = mesh_gpu.edges_packed()
            test_input_gpu = torch.randn(edges_gpu.shape[0], 3, device=device)
            output_gpu = gather_scatter(test_input_gpu, edges_gpu, directed=False, backward=False)
            print("✅ gather_scatter operation works on GPU")
            
            # Verify results are on correct device
            if output_gpu.device.type == 'cuda':
                print("✅ PyTorch3D operations correctly use GPU")
            else:
                print("⚠️  PyTorch3D operations may be falling back to CPU")
            
        except Exception as e:
            print(f"❌ Graph convolution operations failed: {e}")
            return False
        
        print("\n🎉 SUCCESS: PyTorch3D has full CUDA support!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import PyTorch3D: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch3D CUDA test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pytorch3d_cuda()
    sys.exit(0 if success else 1)
