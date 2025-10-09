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
    # Get the packed list of all unique edges in the batch of meshes.
    edges = meshes.edges_packed()
    # Get the packed tensor of all vertices in the batch of meshes.
    verts = meshes.verts_packed()
    # For each edge, retrieve the coordinates of its two endpoint vertices.
    v0, v1 = verts[edges].unbind(1)
    # Compute the squared Euclidean length of each edge.
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


def mesh_edge_loss_highdim(meshes, vert_features, target_length: float = 0.0):
    """
    Modified version of pytorch3d.loss.mesh_edge_loss which bases the edge
    lengths on a tensor of vertex features.

    Args:
        meshes: Meshes object with a batch of meshes.
        vert_features: (sum(V_n), D) with vertex features.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = vert_features  # (sum(V_n), D)
    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODO (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.
    weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / weights.float()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    loss = ((v0 - v1).norm(dim=1, p=2) - target_length) ** 2.0
    loss = loss * weights

    return loss.sum() / N

def mesh_laplacian_loss_highdim(meshes, vert_features, p=2):
    """
    Computes a high-dimensional Laplacian loss on vertex features.

    Args:
        meshes: Meshes object containing the mesh structure.
        vert_features: (sum(V_n), D) tensor of vertex features.
        p: The order of the Laplacian (number of times to apply the Laplacian operator).

    Returns:
        loss: Scalar Laplacian loss value.
    """
    assert p >= 1  # Ensure the Laplacian order is at least 1

    # Get the packed Laplacian matrix for all meshes in the batch
    L = meshes.laplacian_packed()

    # Initialize Lx as the vertex features
    Lx = vert_features

    # Apply the Laplacian operator p times
    for i in range(p):
        Lx = L.mm(Lx)

    # Compute the loss as the sum of the elementwise product of vert_features and Lx
    # This is equivalent to a quadratic form: sum_i <x_i, (L^p x_i)>
    loss = torch.sum(vert_features * Lx)

    # Alternative (commented out): sum of squared Laplacian values
    # loss = L.mm(vert_features)
    # loss = torch.sum(loss ** 2)

    return loss
