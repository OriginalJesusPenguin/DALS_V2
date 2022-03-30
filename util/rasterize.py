from typing import Sequence, Optional, Union

import torch
from pytorch3d.renderer import rasterize_meshes
from pytorch3d.structures import Meshes

def rasterize_vol(
    mesh: Meshes,
    shape: Sequence[int],
    dtype: torch.dtype = torch.uint8,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """
    Rasterize a mesh (with normalized coordinates) to a volume.
    The voxels in the output volume have values:
     - 1 if the voxel center is contained in the mesh.
     - 0 if the voxel center is outside the mesh.
     
    It is assumed that the mesh is a closed surface contained in [-1, 1]^3.
    
    Args:
        mesh: Meshes structure to rasterize.
        shape: Shape of output volume in (H, W, D).
        dtype: Torch dtype of output volume. Default: torch.uint8.
        device: Torch device of output volume. Default: device of mesh.
    
    Returns:
        vol: (H, W, D) Tensor of given shape with rasterized mesh.
    """
    if device is None:
        device = mesh.device
    
    # Move the mesh up, since the PyTorch3D renderer assumes the image plane
    # is at z = 0, but the mesh is in [-1, 1] x [-1, 1] x [-1, 1].
    offset = torch.tensor([0, 0, 1 + 1e-6], device=mesh.device)
    mesh = mesh.offset_verts(offset)
    
    # The PyTorch3D renderer always assumes square pixels. Therefore, we need
    # to rescale the mesh so it looks correct in the rendering.
    # See the documentation of rasterize_meshes.
    # TODO: Make custom CUDA kernel which does not have this issue.
    if shape[0] < shape[1]:
        scale = torch.tensor([[1, shape[1] / shape[0], 1]],
                             device=mesh.device)
        mesh.scale_verts_(scale)
    elif shape[0] > shape[1]:
        scale = torch.tensor([[shape[0] / shape[1], 1, 1]],
                             device=mesh.device)
        mesh.scale_verts_(scale)
    
    # Compute the zbuffer which gives the distance from the pixel to the
    # triangles covering that pixel. Note, rasterize_meshes takes the input
    # in (H, W) but shape is (W, H, D). Also, the default of rasterize_meshes
    # is to store at most 8 triangles per pixel. This seems enough but could
    # be an issue for very complex meshes.
    # TODO: Make custom CUDA kernel which does not have this issue.
    zbuffer = rasterize_meshes(mesh, (shape[1], shape[0]))[1]
    
    # Create output volume. We pad it such that if a mesh is slightly out of
    # bounds (e.g. due to numerical issues) it won't crash everything.
    shape_padded = (shape[0], shape[1], shape[2] + 2)
    vol = torch.zeros(shape_padded, dtype=dtype, device=device)
    
    for i in range(zbuffer.shape[-1]):
        zbuf = zbuffer[0, :, :, i]
        has_triangle = zbuf >= 0
        ys, xs = torch.nonzero(has_triangle, as_tuple=True)
        zs = 1 + (zbuf[has_triangle] * 0.5 * (shape[2] - 1)).ceil().long()
        zs.clamp_(min=0, max=shape[2] + 1)
        
        # Mark voxel in output volume as having an intersecting triangle.
        vol[-xs - 1, -ys - 1, zs] += 1
    
    # Cumsum will compute the number of triangles intersected so far for a
    # given ray. Odd counts mean inside, even mean outside.
    return vol.cumsum(dim=2)[:, :, 1:-1] % 2
