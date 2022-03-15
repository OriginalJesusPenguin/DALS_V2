from math import sqrt

import torch
from pytorch3d.structures import Meshes


def mesh_bl_quality_loss(
    meshes: Meshes, 
    reduction: str = 'mean',
    eps=1e-8,
) -> torch.Tensor:
    """
    Mesh triangle quality loss.

    Triangle quality is measured as
        Q = 4 * sqrt(3) * A / (l1 ** 2 + l2 ** 2 + l3 ** 2)
    where A is the triangle area and l1, l2, l3 are the lengths of the
    triangle edges. The loss is 1 - Q, as ideal triangles have Q = 1.
    This measure was introduced in
        R. P. Bhatia & K. L. Lawrence,
        Two-Dimensional Finite Element Mesh Generation Based on Stripwise
        Automatic Triangulation, Computers and Structures, 1990.

    Args:
        meshes: A batch of meshes.
        reduction: Reduction over all triangles. Must be 'mean' or 'sum'.
        eps: Small constant to avoid division by zero.

    Returns:
        loss: Total loss for all triangles in the batch of meshes.
    """
    if reduction not in ['mean', 'sum']:
        raise ValueError(
            f"reduction must be 'mean' or 'sum' but was: {reduction}"
        )
    # TODO: Maybe weigh contribution by each mesh by number of triangles.

    # First, precompute all edge lengths
    edges = meshes.edges_packed()
    verts = meshes.verts_packed()
    v0, v1 = verts[edges].unbind(1)
    sqrd_lengths = torch.sum((v0 - v1) ** 2, dim=-1)

    # Now, compute the loss components for each triangle
    face_to_edges = meshes.faces_packed_to_edges_packed()
    face_areas = meshes.faces_areas_packed()

    # Sum up losses and return
    denoms = sqrd_lengths[face_to_edges].sum(dim=1) + eps
    loss = 4 * sqrt(3) * torch.sum(face_areas / denoms)
    if reduction == 'mean':
        loss /= len(face_areas)

    return 1.0 - loss

