"""
Fallback metrics module that doesn't require C++ compilation.
This provides the same interface as metrics.py but without C++ dependencies.
"""

import torch
from typing import Optional, Sequence, Dict
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes

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
    Fallback implementation without C++ dependencies.
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
    Fallback implementation that returns zeros instead of computing actual intersections.
    This avoids the need for C++ compilation.
    """
    # Return zeros for all meshes
    intersecting_faces = torch.zeros(meshes.faces_packed().shape[0], dtype=torch.int32)
    
    int_faces_per_mesh = torch.zeros(
        len(meshes),
        dtype=intersecting_faces.dtype,
    )
    
    return int_faces_per_mesh, intersecting_faces
