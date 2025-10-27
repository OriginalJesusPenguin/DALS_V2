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


def mesh_jacobian_determinant_loss(template_meshes, deformed_meshes, eps=1e-8):
    """
    Compute Jacobian determinant loss to prevent self-intersections.
    
    This loss penalizes local deformations that flip orientation (det(J) < 0),
    which is associated with self-intersections in mesh deformation.
    
    The Jacobian is computed for each face by looking at how the face deforms.
    For triangular faces, we compute the 2x2 Jacobian in the local face plane.
    
    Args:
        template_meshes: Meshes object with original/template vertices.
        deformed_meshes: Meshes object with deformed vertices.
        eps: Small constant for numerical stability.
        
    Returns:
        loss: Scalar loss value penalizing negative Jacobian determinants.
    """
    if len(template_meshes) == 0 or len(deformed_meshes) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=template_meshes.device)
    
    template_verts = template_meshes.verts_packed()  # (N, 3)
    deformed_verts = deformed_meshes.verts_packed()  # (N, 3)
    
    # Get face connectivity
    faces = template_meshes.faces_packed()  # (F, 3)
    
    if faces.shape[0] == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=template_meshes.device)
    
    # Get vertex positions for each face
    # Template vertices
    v0_template = template_verts[faces[:, 0]]  # (F, 3)
    v1_template = template_verts[faces[:, 1]]  # (F, 3)
    v2_template = template_verts[faces[:, 2]]  # (F, 3)
    
    # Deformed vertices
    v0_deformed = deformed_verts[faces[:, 0]]  # (F, 3)
    v1_deformed = deformed_verts[faces[:, 1]]  # (F, 3)
    v2_deformed = deformed_verts[faces[:, 2]]  # (F, 3)
    
    # Compute edge vectors in template (local basis)
    e1_template = v1_template - v0_template  # (F, 3)
    e2_template = v2_template - v0_template  # (F, 3)
    
    # Compute edge vectors in deformed mesh
    e1_deformed = v1_deformed - v0_deformed  # (F, 3)
    e2_deformed = v2_deformed - v0_deformed  # (F, 3)
    
    # Compute face normals (used as third basis vector)
    # Template normal
    n_template = torch.cross(e1_template, e2_template, dim=1)  # (F, 3)
    n_template_norm = torch.norm(n_template, dim=1, keepdim=True) + eps
    n_template = n_template / n_template_norm  # (F, 3)
    
    # Deformed normal
    n_deformed = torch.cross(e1_deformed, e2_deformed, dim=1)  # (F, 3)
    n_deformed_norm = torch.norm(n_deformed, dim=1, keepdim=True) + eps
    n_deformed = n_deformed / n_deformed_norm  # (F, 3)
    
    # Project edges onto face plane (for computing in-plane Jacobian)
    # For 2D Jacobian in the face plane, we need to project onto local 2D coordinates
    # Build local orthonormal basis for each face
    
    # Basis 1: normalized e1
    basis1_template = e1_template / (torch.norm(e1_template, dim=1, keepdim=True) + eps)  # (F, 3)
    
    # Basis 2: e2 projected onto plane perpendicular to basis1
    # Gram-Schmidt process
    proj = torch.sum(e2_template * basis1_template, dim=1, keepdim=True)  # (F, 1)
    basis2_template = e2_template - proj * basis1_template  # (F, 3)
    basis2_template = basis2_template / (torch.norm(basis2_template, dim=1, keepdim=True) + eps)  # (F, 3)
    
    # Same for deformed
    basis1_deformed = e1_deformed / (torch.norm(e1_deformed, dim=1, keepdim=True) + eps)  # (F, 3)
    proj = torch.sum(e2_deformed * basis1_deformed, dim=1, keepdim=True)  # (F, 1)
    basis2_deformed = e2_deformed - proj * basis1_deformed  # (F, 3)
    basis2_deformed = basis2_deformed / (torch.norm(basis2_deformed, dim=1, keepdim=True) + eps)  # (F, 3)
    
    # Project edges onto local 2D coordinates
    e1_2d_template = torch.stack([
        torch.sum(e1_template * basis1_template, dim=1),  # (F,)
        torch.sum(e1_template * basis2_template, dim=1)    # (F,)
    ], dim=1)  # (F, 2)
    
    e2_2d_template = torch.stack([
        torch.sum(e2_template * basis1_template, dim=1),  # (F,)
        torch.sum(e2_template * basis2_template, dim=1)    # (F,)
    ], dim=1)  # (F, 2)
    
    e1_2d_deformed = torch.stack([
        torch.sum(e1_deformed * basis1_deformed, dim=1),  # (F,)
        torch.sum(e1_deformed * basis2_deformed, dim=1)    # (F,)
    ], dim=1)  # (F, 2)
    
    e2_2d_deformed = torch.stack([
        torch.sum(e2_deformed * basis1_deformed, dim=1),  # (F,)
        torch.sum(e2_deformed * basis2_deformed, dim=1)    # (F,)
    ], dim=1)  # (F, 2)
    
    # Stack to form 2x2 Jacobian matrices for each face
    # J = [deformed_edges] @ inv([template_edges])
    # For numerical stability, compute using cross products in 2D
    # 
    # For each face i, we have:
    # J_i = [e1_deformed_i  e2_deformed_i] @ inv([e1_template_i  e2_template_i])
    # 
    # We can compute the determinant of J directly using:
    # det(J) = det([deformed]) / det([template])
    # 
    # For 2x2 matrices [a b; c d], det = ad - bc
    
    # Compute determinant of template edges (denominator)
    # det_template = e1_template_x * e2_template_y - e1_template_y * e2_template_x
    det_template = e1_2d_template[:, 0] * e2_2d_template[:, 1] - \
                   e1_2d_template[:, 1] * e2_2d_template[:, 0]
    det_template = det_template + eps  # Avoid division by zero
    
    # Compute determinant of deformed edges (numerator)
    det_deformed = e1_2d_deformed[:, 0] * e2_2d_deformed[:, 1] - \
                   e1_2d_deformed[:, 1] * e2_2d_deformed[:, 0]
    
    # Compute ratio
    all_dets = det_deformed / det_template
    
    # Penalize negative determinants (orientation flips)
    # Using ReLU to only penalize negative values
    neg_loss = torch.relu(-all_dets)
    
    # Also add a small penalty for very large determinants (extreme scaling)
    max_det = torch.relu(all_dets - 1.0)  # Penalize det > 1.0
    
    loss = neg_loss.mean() + 0.1 * max_det.mean()
    
    return loss
