from typing import Optional, Sequence, Dict

import torch
from torch.utils.cpp_extension import load

from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes

_C_tri = load('_C_tri', ['util/csrc/triangle_self_intersections.cpp'],
              extra_cflags=['-O3', '-march=native'], verbose=True)

def point_metrics(
    points_pred: torch.Tensor,
    points_true: torch.Tensor,
    thresholds: Sequence[float],
    lengths_pred: Optional[torch.Tensor] = None,
    lengths_true: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute precision, recall, and F1-score.

    Args:
        points_pred: (N, P1, 3) tensor with predicted points.
        points_true: (N, P2, 3) tensor with true points.
        lengths_pred: (N,) tensor with lengths of predicted point sets in the
            batch or None if all lengths are P1.
        lengths_true: (N,) tensor with lengths of true point sets in the batch
            or None if all lengths are P2.
        thresholds: Distance thresholds for scores.
        eps: Small constant for avoiding division by zero.

    Returns:
        metrics: Dictionary with metrics.
    """
    knn_pred = knn_points(points_pred, points_true,
                          lengths1=lengths_pred, lengths2=lengths_true)
    pred_to_true_dists = knn_pred.dists[:, :, 0].sqrt()
    knn_true = knn_points(points_true, points_pred,
                          lengths1=lengths_true, lengths2=lengths_pred)
    true_to_pred_dists = knn_true.dists[:, :, 0].sqrt()

    metrics = dict()
    for t in thresholds:
        precision = 100.0 * (pred_to_true_dists < t).float().mean(dim=1)
        recall = 100.0 * (true_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)

        t_str = f'{t:f}'.rstrip('0')  # 'f' format leaves trailing zeros
        metrics['Precision@' + t_str] = precision
        metrics['Recall@' + t_str] = recall
        metrics['F1@' + t_str] = f1
    metrics['Hausdorff'] = max(pred_to_true_dists.max(),
                               true_to_pred_dists.max())

    # Ensure metrics are on the CPU
    metrics = { k: v.cpu() for k, v in metrics.items() }

    return metrics


def self_intersections(meshes: Meshes):
    """
    Count triangle self intersections for meshes.

    Uses the algorithm from:
        Moller, A Fast Triangle-Triangle Test, Journal of Graphics Tools. 1997
    as implemented at:
        http://web.archive.org/web/19990203013328/http://www.acm.org/jgt/papers/Moller97/tritri.html

    Args:
        meshes: Meshes structure with batch of meshes

    Returns:
        int_faces_per_mesh: Tensor with number of intersections for each mesh
        intersecting_faces: (sum(F_n),) Tensor with no. intersections for each
            face.
    """
    verts = meshes.verts_packed()
    # Seems like Torch's C++ API does not like 64-bit integers. So, we convert
    # to 32-bit since that's enough and will make Torch happy.
    faces = meshes.faces_packed().to(torch.int32)

    intersecting_faces = _C_tri.triangle_self_intersections(verts, faces)

    int_faces_per_mesh = torch.zeros(
        len(meshes),
        dtype=intersecting_faces.dtype,
    )
    int_faces_per_mesh.scatter_add_(
        0, meshes.faces_packed_to_mesh_idx(), intersecting_faces
    )

    return int_faces_per_mesh, intersecting_faces
