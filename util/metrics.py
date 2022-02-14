from typing import Optional, Sequence, Dict

import torch
from pytorch3d.ops import knn_points


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
        
    # Ensure metrics are on the CPU
    metrics = { k: v.cpu() for k, v in metrics.items() }

    return metrics
