import re
from typing import Tuple, Union, Iterable
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCELoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        """
        Parameters:
            reduction: str with reduction to use. Must be one of:
                * 'mean': Return mean over annotated planes.
                * 'sum': Return sum over annotated planes.
                * 'none': Don't perform any reduction.
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.reduction = reduction


    def forward(self, x, target, mask):
        """
        Compute masked cross entropy loss.

        Parameters:
            x: (N, C, W, H, D) Tensor with predictions.
            target: (N, W, H, D) Tensor with target values.
            mask: (N, W, H, D) Tensor specifying voxels with labels.

        Returns:
            loss: scalar or N x W x H x D Tensor with loss value.
        """
        loss = self.ce_loss(x, target * mask)
        loss *= mask
        if self.reduction == 'mean':
            return loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # self.reduction == 'none'
            return loss


class SliceCELoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        """
        Parameters:
            reduction: str with reduction to use. Must be one of:
                * 'mean': Return mean over annotated planes.
                * 'sum': Return sum over annotated planes.
                * 'none': Don't perform any reduction.
        """
        super().__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.masked_ce_loss = MaskedCELoss(reduction=reduction)


    def forward(self, x: torch.Tensor, target: torch.Tensor,
                x_coords=None, y_coords=None, z_coords=None) -> torch.Tensor:
        """
        Compute slice cross entropy loss.

        Parameters:
            x: (N, C, W, H, D) Tensor with predictions
            target: (N, W, H, D) Tensor with target values
            x_coords: None or array-like with x-coords for yz-slices.
            y_coords: None or array-like with y-coords for xz-slices.
            z_coords: None or array-like with z-coords for xy-slices.

        Returns:
            loss: scalar or N x W x H x D Tensor with loss value.
        """
        # Build slice mask
        mask = torch.zeros_like(target)
        if x_coords is not None:
            for i, xs in enumerate(x_coords):
                mask[i, xs, :, :] = 1
        if y_coords is not None:
            for i, ys in enumerate(y_coords):
                mask[i, :, ys, :] = 1
        if z_coords is not None:
            for i, zs in enumerate(z_coords):
                mask[i, :, :, zs] = 1

        # Compute loss
        return self.masked_ce_loss(x, target, mask)


class SoftIntersectionLoss(nn.Module):
    """
    Computes a soft plane intersection loss.

    The soft intersection loss is given by the softmin of squared distances
    between a set of points and a plane. If the object the points represent
    intersects the plane the softmin should be (close to) 0.

    To make the loss curve flatter in the intersection region an alpha factor
    is multiplied on the squared distances before softmin:

        softmin(x) = exp(-alpha * x) / sum(exp(-alpha * x)).

    A large alpha > 1 makes the softmin closer to a the real min. However, it
    can also cause numerical instabilities for large distances.
    """

    def __init__(self, alpha: float = 1.0, squared_dists: bool = True):
        """
        Parameters:
            alpha: float with alpha factor for softmin.
            squared_dists: If True, true use squared point-to-plane distances.
        """
        super().__init__()
        self.alpha = alpha
        self.squared_dists = squared_dists


    def intersection_loss(
        self,
        points: torch.Tensor,
        plane_normal,
        plane_dist
    ) -> torch.Tensor:
        """
        Compute soft intersection loss between points and a single plane.

        Parameters:
            points: (V_n, 3) Tensor with point coordinates.
            plane_normal: Length 3 array_like with plane normal.
            plane_dist: Scalar with distance from origin to plane.

        Returns:
            loss: Scalar tensor with loss value
        """
        # Ensure plane normal is a normalized torch Tensor
        # NOTE: We create a new tensor to avoid errors with inplace ops later
        plane_normal = torch.tensor(plane_normal,
                                    dtype=points.dtype,
                                    device=points.device)
        plane_normal /= plane_normal.norm()

        # Compute point to plane distances
        point_plane_dists = torch.mv(points, plane_normal) - plane_dist
        if self.squared_dists:
            point_plane_dists = point_plane_dists ** 2
        else:
            point_plane_dists.abs_()

        sm = F.softmin(point_plane_dists * self.alpha)
        return torch.sum(point_plane_dists * sm)


    def forward(
        self,
        points: torch.Tensor,
        plane_normals,
        plane_dists
    ) -> torch.Tensor:
        """
        Compute soft intersection loss between points and multiple planes.

        Parameters:
            points: (V_n, 3) Tensor with point coordinates.
            plane_normals: (P_n, 3) array-like with plane normals.
            plane_dists: Length P_n array-like with distances from origin to
                planes.

        Returns:
            loss: Scalar tensor with sum of intersection loss for each plane.
        """
        return sum(self.intersection_loss(points, n, d)
                   for n, d in zip(plane_normals, plane_dists))


def mesh_plane_intersection(
    verts: torch.Tensor,
    faces: torch.Tensor,
    plane_normal,
    plane_dist,
    return_points: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute intersection between triangle mesh and plane.

    Intersection points are returned as barycentric coordinates for each
    intersecting face.

    Parameters:
        verts: (V_n, 3) Tensor with vertex coordinates.
        faces: (F_n, 3) Tensor with face indices.
        plane_normal: Length 3 array_like with plane normal.
        plane_dist: Scalar with distance from origin to plane.
        return_points: If True, also return coordinates of intersection points.

    Returns:
        bary_coords: (N, 2, 3) FloatTensor with barycentric coordinates.
        face_idxs: (N, 1) LongTensor with intersecting triangles.
    """
    # Cases from https://github.com/mikedh/trimesh/blob/master/trimesh/intersections.py#L57
    sign_cases = torch.as_tensor([
        [-1, -1, -1],  # 0 : No
        [-1, -1,  0],  # 1 : No
        [-1, -1,  1],  # 2 : Yes, 2 on one side, 1 on the other
        [-1,  0,  0],  # 3 : Yes, edge on plane (but ignore to avoid repeats)
        [-1,  0,  1],  # 4 : Yes, 1 one plane, 2 on different sides
        [-1,  1,  1],  # 5 : Yes, 2 on one side, 1 on the other
        [ 0,  0,  0],  # 6 : No, triangle on plane
        [ 0,  0,  1],  # 7 : Yes, edge on plane
        [ 0,  1,  1],  # 8 : No
        [ 1,  1,  1],  # 9 : No
    ], device=verts.device)

    # Find which side of the plane vertices are on
    # NOTE: We create a new tensor to avoid errors with inplace ops later
    plane_normal = torch.tensor(plane_normal,
                                dtype=verts.dtype,
                                device=verts.device)
    plane_normal /= plane_normal.norm()
    vert_dots = torch.mv(verts, plane_normal) - plane_dist
    face_signs = torch.sign(vert_dots)[faces]
    face_signs_sorted = face_signs.sort(dim=1)[0]

    # Check which faces intersect the plane and how
    edge_on_plane = face_signs_sorted.eq(sign_cases[7]).all(dim=1)
    vert_on_plane = face_signs_sorted.eq(sign_cases[4]).all(dim=1)
    basic = face_signs_sorted.eq(sign_cases[2]).all(dim=1) \
          | face_signs_sorted.eq(sign_cases[5]).all(dim=1)


    def find_zero_comb(p1, p2):
        # Find alpha such that alpha * p1 + (1 - alpha) * p2 = 0
        return -p2 / (p1 - p2)


    def handle_edge_on_plane(face_signs):
        device = face_signs.device
        bary_coords = torch.zeros((len(face_signs), 2, 3),
                                  device=device).float()
        if len(face_signs) == 0:
            return bary_coords  # Bail early!
        index = torch.as_tensor([[0, 1, 2]], device=device).expand(
            len(face_signs), -1)
        face_vert_idxs = index[face_signs == 0].reshape(-1, 2)
        _fill_rows(bary_coords[:, 0, :], face_vert_idxs[:, 0], 1.0)
        _fill_rows(bary_coords[:, 1, :], face_vert_idxs[:, 1], 1.0)
        return bary_coords


    def handle_vert_on_plane(face_vert_dots):
        device = face_vert_dots.device
        bary_coords = torch.zeros((len(face_vert_dots), 2, 3),
                                  device=device).float()
        if len(face_vert_dots) == 0:
            return bary_coords  # Bail early!
        index = torch.as_tensor([[0, 1, 2]], device=device).expand(
            len(face_vert_dots), -1)
        plane_vert_idxs = index[face_vert_dots == 0]
        opposite_edges = index[face_vert_dots != 0].reshape(-1, 2)
        edge_vert_dots = face_vert_dots.gather(1, opposite_edges)
        alphas = find_zero_comb(edge_vert_dots[:, 0], edge_vert_dots[:, 1])
        _fill_rows(bary_coords[:, 0, :], plane_vert_idxs, 1.0)
        _set_rows(bary_coords[:, 1, :], opposite_edges[:, 0], alphas)
        _set_rows(bary_coords[:, 1, :], opposite_edges[:, 1], 1.0 - alphas)
        return bary_coords


    def handle_basic(face_vert_dots, face_signs):
        device = face_vert_dots.device
        bary_coords = torch.zeros((len(face_vert_dots), 2, 3),
                                  device=device).float()
        if len(face_vert_dots) == 0:
            return bary_coords  # Bail early!
        index = torch.as_tensor([[0, 1, 2]], device=device).expand(
            len(face_vert_dots), -1)
        unique_is_neg = face_signs.sum(dim=1) > 0
        unique_verts = index[_rows_where(unique_is_neg,
                                         face_vert_dots < 0,
                                         face_vert_dots > 0)]
        opposite_edges = index[_rows_where(unique_is_neg,
                                           face_vert_dots > 0,
                                           face_vert_dots < 0)].reshape(-1, 2)
        edge_vert_dots = face_vert_dots.gather(1, opposite_edges)
        unique_vert_dots = face_vert_dots.gather(
            1, unique_verts.unsqueeze(1)).squeeze()
        alphas0 = find_zero_comb(unique_vert_dots, edge_vert_dots[:, 0])
        alphas1 = find_zero_comb(unique_vert_dots, edge_vert_dots[:, 1])
        _set_rows(bary_coords[:, 0, :], unique_verts, alphas0)
        _set_rows(bary_coords[:, 0, :], opposite_edges[:, 0], 1.0 - alphas0)
        _set_rows(bary_coords[:, 1, :], unique_verts, alphas1)
        _set_rows(bary_coords[:, 1, :], opposite_edges[:, 1], 1.0 - alphas1)
        return bary_coords


    bary_coords = torch.cat([
        handle_edge_on_plane(face_signs[edge_on_plane]),
        handle_vert_on_plane(vert_dots[faces[vert_on_plane]]),
        handle_basic(vert_dots[faces[basic]], face_signs[basic])
    ], dim=0)
    face_idxs = torch.cat([
        edge_on_plane.nonzero(), vert_on_plane.nonzero(), basic.nonzero()
    ]).squeeze()

    if not return_points:
        return bary_coords, face_idxs
    else:
        # Get vertex coordinates of faces.
        face_verts = verts[faces]
        fv0, fv1, fv2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        # Get vertex coordinates of intersection curve.
        a = fv0[face_idxs]
        b = fv1[face_idxs]
        c = fv2[face_idxs]
        v0, v1 = _apply_int_bary_coords(a, b, c, bary_coords)
        int_points = torch.cat([v0[:,None,:], v1[:,None,:]], dim=1)
        return bary_coords, face_idxs, int_points


def sample_points_from_intersection(
    verts: torch.Tensor,
    faces: torch.Tensor,
    plane_normal,
    plane_dist,
    num_samples: int = 10000,
    return_normals: bool = False,
    return_indices: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sample points uniformly on intersection between mesh and plane.

    Parameters:
        verts: (V_n, 3) Tensor with vertex coordinates.
        faces: (F_n, 3) Tensor with face indices.
        plane_normal: Length 3 array_like with plane normal.
        plane_dist: Scalar with distance from origin to plane.
        num_samples: Integer with the number of point to sample.
        return_normals: If True, return in-plane normals for intersecting
            faces.

    Returns:
        samples: (num_samples, 3) FloatTensor with sample points. If the
            intersection is empty, a (0, 3) FloatTensor is returned.
        normals: (num_samples, 3) FloatTensor with sample point in-plane
            normals. If the intersection is empty, a (0, 3) FloatTensor is
            returned. Only returned if return_normals is True.
    """
    # Compute intersection between mesh and plane.
    with torch.no_grad():
        # Ensure plane_normal is normalized and a PyTorch tensor
        plane_normal = torch.as_tensor(plane_normal,
                                       dtype=verts.dtype,
                                       device=verts.device)
        plane_normal /= plane_normal.norm()

        bary_coords, int_face_idxs = mesh_plane_intersection(
            verts, faces, plane_normal, plane_dist)

    if len(bary_coords) == 0:
        # Mesh and plane do not intersect so return empty arrays
        if return_normals:
            return torch.zeros(0, 3, device=verts.device), \
                   torch.zeros(0, 3, device=verts.device)
        else:
            return torch.zeros(0, 3, device=verts.device)

    # Get vertex coordinates of faces.
    face_verts = verts[faces]
    fv0, fv1, fv2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Get vertex coordinates of intersection curve.
    a = fv0[int_face_idxs]
    b = fv1[int_face_idxs]
    c = fv2[int_face_idxs]

    v0, v1 = _apply_int_bary_coords(a, b, c, bary_coords)

    # Compute number of samples on each line segment of intersection curve.
    with torch.no_grad():
        lengths = torch.norm(v1 - v0, p=2, dim=1)
        sample_line_idxs = lengths.multinomial(num_samples, replacement=True)
        sample_to_face_idxs = int_face_idxs[sample_line_idxs]

    # Compute samples.
    alphas = torch.rand(num_samples, 1, device=verts.device)
    samples = alphas * v0[sample_line_idxs] \
            + (1 - alphas) * v1[sample_line_idxs]

    # TODO: Find a more clean version to deal with return options
    if return_normals:
        # Compute (non-normalized) face normals
        face_normals = torch.cross(b - a, c - a, dim=1)

        # Get sample normals and project onto plane
        normals = face_normals[sample_line_idxs]
        normals -= normals.mv(plane_normal).unsqueeze(1) * plane_normal
        normals = normals / normals.norm(p=2, dim=1, keepdim=True).clamp(
            min=sys.float_info.epsilon)
        if return_indices:
            return samples, normals, int_face_idxs, sample_to_face_idxs
        else:
            return samples, normals
    else:
        if return_indices:
            return samples, int_face_idxs, sample_to_face_idxs
        else:
            return samples


def sample_points_from_intersections(
    verts: torch.Tensor,
    faces: torch.Tensor,
    plane_normals,
    plane_dists,
    num_samples_per_plane: Union[int, Iterable[int]] = 10000,
    return_point_weights: bool = True,
    return_normals: bool = True,
    return_as_list: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sample points uniformaly on intersections between mesh and planes.

    Parameters:
        verts: (V_n, 3) Tensor with vertex coordinates.
        faces: (F_n, 3) Tensor with face indices.
        plane_normals: (P_n, 3) array-like with plane normals.
        plane_dists: Length P_n array-like with distances from origin to
            planes.
        num_samples_per_plane: Integer or length P_n array-like with number of
            points to sample from each plane.
        return_point_weights: If true, return weight for each point equal to
            1 / number planes which intersecton the triangle the point is in.
        return_normals: If True, return in-plane normals for intersecting
            faces.
        return_as_list: If True, return a list of samples for each plane. If
            False, return a single tensor with all samples.

    Returns:
        If return_as_list is False (default):
            samples: (total_samples, 3) FloatTensor with sample points.
            normals: (total_samples, 3) FloatTensor with point in-plane
                normals. Only returned if return_normals is True.
        If return_as_list is True:
            samples: List of (num_samples_per_plane[i], 3) FloatTensors with
                sample points. If the i'th intersection was empty the i'th
                entry will be a (0, 3) FloatTensor..
            normals: List of (num_samples_per_plane, 3) FloatTensors with
                sample point in-plane normals. If the i'th intersection was
                empty the i'th entry will be (0,3) FloatTensor. Only returned
                if return_normals is True.
    """
    # Validate input.
    assert len(plane_normals) == len(plane_dists), \
        "Must supply a distance for each plane normal"

    if not hasattr(num_samples_per_plane, '__iter__'):
        num_samples_per_plane = [num_samples_per_plane] * len(plane_normals)

    # Compute samples.
    results = defaultdict(list)
    parts = zip(plane_normals, plane_dists, num_samples_per_plane)
    for normal, dist, num_samples in parts:
        res = sample_points_from_intersection(
            verts, faces, normal, dist, num_samples,
            return_normals=return_normals,
            return_indices=return_point_weights
        )
        # Ensure res is tuple for convenience
        if not isinstance(res, tuple):
            res = (res,)
        results['samples'].append(res[0])
        if return_normals:
            results['normals'].append(res[1])
        if return_point_weights:
            results['int_face_idxs'].append(res[-2])
            results['sample_to_face_idxs'].append(res[-1])

    # Compute point weights if needed
    if return_point_weights:
        face_counts = torch.zeros(len(faces), device=faces.device)


    # Unpack and return results.
    samples = results['samples']
    if return_normals:
        normals = results['samples']
        if return_as_list:
            return samples, normals
        else:
            return torch.cat(samples, dim=0), torch.cat(normals, dim=0)
    else:
        return samples if return_as_list else torch.cat(samples, dim=0)


def _apply_int_bary_coords(a, b, c, bary_coords):
    # Unpack barycentric coords.
    w00 = bary_coords[:, 0, 0].unsqueeze(1)
    w01 = bary_coords[:, 0, 1].unsqueeze(1)
    w02 = bary_coords[:, 0, 2].unsqueeze(1)
    w10 = bary_coords[:, 1, 0].unsqueeze(1)
    w11 = bary_coords[:, 1, 1].unsqueeze(1)
    w12 = bary_coords[:, 1, 2].unsqueeze(1)

    # Compute intersection points
    v0 = w00 * a + w01 * b + w02 * c
    v1 = w10 * a + w11 * b + w12 * c

    return v0, v1


def _set_rows(x, indices, values):
    assert len(x) == len(indices)
    assert len(x) == len(values)
    for i in range(x.shape[1]):
        x[indices == i, i] = values[indices == i]
    return x


def _fill_rows(x, indices, value):
    assert len(x) == len(indices)
    for i in range(x.shape[1]):
        x[indices == i, i] = value
    return x


def _rows_where(condition, x, y):
    # Return tensor out where out[i] = x[i] if condition[i] else y[i].
    assert x.shape == y.shape
    out = torch.zeros_like(x)
    out[condition] = x[condition]
    out[~condition] = y[~condition]
    return out

