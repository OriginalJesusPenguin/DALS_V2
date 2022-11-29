from typing import Tuple, Optional, Sequence

import numpy as np
from scipy.spatial import ConvexHull
import torch
from pytorch3d.structures import Meshes
from pytorch3d._C import point_face_dist_forward

from util.bin.pyremesh import remesh_botsch

def split_and_collapse_sphere(mesh, sphere_template):
    edges = mesh.edges_packed()
    mverts = mesh.verts_packed()
    tverts = sphere_template.verts_packed()
    edge_lengths = torch.norm(mverts[edges[:, 1]] - mverts[edges[:, 1]],
                              dim=1)

    mean_length = edge_lengths.mean()
    v0, v1 = tverts[edges[:, 0]], tverts[edges[:, 1]]

    split_idx = edge_lengths > 4/3 * mean_length
    new_verts_split = 0.5 * (v0[split_idx] + v1[split_idx])
    new_verts_split /= new_verts_split.norm(dim=1, keepdim=True)

    collapse_idx = edge_lengths < 4/5 * mean_length
    keep_vert_idx = torch.full((len(tverts),), True)
    keep_vert_idx[edges[collapse_idx].flatten()] = False
    new_verts_collapse = 0.5 * (v0[collapse_idx] + v1[collapse_idx])
    new_verts_collapse /= new_verts_collapse.norm(dim=1, keepdim=True)

    ch = ConvexHull(torch.cat([
        tverts[keep_vert_idx],
        new_verts_split,
        new_verts_collapse,
    ]).cpu())

    return Meshes(
        [torch.from_numpy(ch.points).float()],
        [torch.from_numpy(ch.simplices)],
    )


def remesh_bk(
    mesh: Meshes,
    target_length: Optional[float] = None,
    target_length_ratio: Optional[float] = None,
    iters: int = 5,
) -> Meshes:
    """
    Remesh using Botsch-Kobbel remeshing

    Args:
        mesh: Meshes structure with mesh to remesh.
        target_length: Target edge length. Default is mean edge length of input
            mesh. Must not be provided if target_length_ratio is provided.
        target_length_ratio: Ratio of target edge length w.r.t. to mean edge
            length of input mesh. Must not be provided if target_length is
            provided.
        iters: Number of resmeshing iterations.

    Returns:
        Meshes structure with remeshed mesh.
    """
    assert target_length is None or target_length_ratio is None, \
        "Cannot specify length ratio if length is also provided"
    device = mesh.device
    verts = mesh.verts_packed().detach()
    faces = mesh.faces_packed().detach()
    if target_length is None:
        v0, v1 = verts[mesh.edges_packed()].unbind(1)
        target_length = torch.norm(v1 - v0, dim=1).mean().cpu()
    if target_length_ratio is None:
        target_length_ratio = 1.0
    new_verts, new_faces = remesh_botsch(
        verts.cpu().numpy().astype(np.float64),
        faces.cpu().numpy().astype(np.int32),
        iters,
        target_length * target_length_ratio,
    )
    new_verts = torch.from_numpy(new_verts).float().to(device)
    new_faces = torch.from_numpy(new_faces).to(device)
    return Meshes([new_verts], [new_faces])


def _interp_bary(vert_feats, faces, vert_to_tri_idx, bary_coords):
    """
    Interpolate vertex features given barycentric coordinates.

    Args:
        vert_feats: (V, D) Tensor with vertex features.
        faces: (F, 3) Tensor with vertex indices for each face.
        vert_to_tri_idx: (W,) Tensor with paired face for each new vertex.
        bary_coords: (W, 3) Tensor with barycentric coordinates.

    Returns:
        (W, D) Tensor with interpolated vertex features
    """
    tri_vert_feats = vert_feats[faces][vert_to_tri_idx]
    new_features = bary_coords[:, 0].unsqueeze(1) * tri_vert_feats[:, 0, :] \
                 + bary_coords[:, 1].unsqueeze(1) * tri_vert_feats[:, 1, :] \
                 + bary_coords[:, 2].unsqueeze(1) * tri_vert_feats[:, 2, :]
    return new_features


def remesh_template_from_deformed(
    deformed_template: Meshes,
    template: Meshes,
    ratio: float = 1.0,
    bk_iters: int = 5,
    sphere_template: bool = True,
    vert_features: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[Meshes, torch.Tensor]:
    """
    Args:
        deformed_template: Meshes structure with deformed template.
        template: Meshes structure with original template.
        ratio: Target edge length ratio w.r.t. mean deformed edge length.
        bk_iters: Number of iteration of Botsch-Kobbelt remeshing to run.
        sphere_template: Ensure all new vertices have the same length as the
            original template.
        vert_features: Additional template vertex features.

    Returns:
        If vert_features is None
            Meshes structure with remeshed template.
        Else
            Meshes structure with remeshed template.
            List with interpolated vertex features.
    """
    # Input validation
    device = template.device
    assert deformed_template.device == device, \
        "Input meshes must be on same device"
    if vert_features is not None:
        for vf in vert_features:
            assert vf.device == device, \
                "Vertex features must be on same device as template"

    # Remesh the deformed mesh with Botsch-Kobbelt remesher
    verts = deformed_template.verts_packed().detach()
    faces = deformed_template.faces_packed().detach()
    v0, v1 = verts[deformed_template.edges_packed()].unbind(1)
    h = torch.norm(v1 - v0, dim=1).mean().cpu()
    new_verts, new_faces = remesh_botsch(
        verts.cpu().numpy().astype(np.float64),
        faces.cpu().numpy().astype(np.int32),
        bk_iters,
        h * ratio,
    )
    new_verts = torch.from_numpy(new_verts).float().to(device)
    new_faces = torch.from_numpy(new_faces).to(device)

    # Find closest triangle in deformed mesh for each new vertex
    # TODO: Might be speed improvements to get from custom kernel
    # ...also, pytorch3d might change this function to not return what we need
    tri_verts = verts[faces]
    _, closest_tri_idx = point_face_dist_forward(
        new_verts, torch.tensor([0], device=device),
        tri_verts, torch.tensor([0], device=device),
        len(new_verts),
    )

    m_tri_verts = tri_verts[closest_tri_idx]

    # Compute barycentric coordinates for each point-triangle pair
    u = m_tri_verts[:, 1, :] - m_tri_verts[:, 0, :]
    v = m_tri_verts[:, 2, :] - m_tri_verts[:, 0, :]
    w = new_verts - m_tri_verts[:, 0, :]

    n = torch.cross(u, v, dim=1)
    n_dot_n = torch.sum(n * n, dim=1)
    b3 = torch.sum(torch.cross(u, w, dim=1) * n, dim=1) / n_dot_n
    b2 = torch.sum(torch.cross(w, v, dim=1) * n, dim=1) / n_dot_n
    b1 = 1 - b2 - b3

    bary = torch.stack([b1, b2, b3], dim=1)

    # Use barycentric coordinates to project onto triangle
    proj_bary = bary.clamp(min=0, max=1)
    proj_bary /= proj_bary.sum(dim=1, keepdim=True)

    # Use projecting barycentric coordinates to create new template vertices
    # and latent vectors
    tverts = template.verts_packed()
    new_template_verts = _interp_bary(
        tverts,
        faces,
        closest_tri_idx,
        proj_bary,
    )
    if sphere_template:
        new_template_verts /= new_template_verts.norm(dim=1, keepdim=True)
        new_template_verts *= tverts.norm(dim=1).mean()

    # Return results
    remeshed_template = Meshes([new_template_verts], [new_faces])
    if vert_features is None:
        return remeshed_template
    else:
        new_features = []
        for vf in vert_features:
            new_features.append(_interp_bary(
                vf,
                faces,
                closest_tri_idx,
                proj_bary,
            ))
        return remeshed_template, new_features