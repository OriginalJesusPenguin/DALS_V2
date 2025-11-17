#!/usr/bin/env python3
"""
Command-line entry point to exercise latent metric utilities.

Example:
    python -m analysis.run_latent_metrics --latent-status healthy --latent-index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.latent_metrics import (  # noqa: E402
    baseline_latent_vector,
    compute_deformation_metrics,
    decoder_from_checkpoint,
    decode_latent,
    finite_difference_latent_sensitivity,
    load_checkpoint,
    load_latent_tensor,
    pullback_metric,
    resolve_latent_path,
    spectral_metrics,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_ckpt = project_root / "models" / "MeshDecoderTrainer_2025-11-06_12-00-26.ckpt"
    default_latent_root = (
        project_root
        / "inference_results"
        / "meshes_MeshDecoderTrainer_2025-11-06_12-00-26"
        / "latents"
    )

    parser = argparse.ArgumentParser(description="Run latent mesh analysis metrics.")
    parser.add_argument("--checkpoint", type=Path, default=default_ckpt)
    parser.add_argument("--latent-path", type=Path, default=None)
    parser.add_argument("--latent-root", type=Path, default=default_latent_root)
    parser.add_argument(
        "--latent-status",
        choices=("healthy", "cirrhotic"),
        default="cirrhotic",
        help="Latent cohort to sample when --latent-path is omitted.",
    )
    parser.add_argument(
        "--latent-index",
        type=int,
        default=0,
        help="Index within the selected latent cohort.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--spectral-k", type=int, default=15)
    parser.add_argument(
        "--spectral-weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional weights for spectral distance (length must match --spectral-k).",
    )
    parser.add_argument(
        "--spectral-times",
        type=float,
        nargs="+",
        default=None,
        help="Optional time samples for HKS (defaults to geometric progression).",
    )
    parser.add_argument("--fd-epsilon", type=float, default=1e-2)
    parser.add_argument(
        "--fd-indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional latent indices for finite differences (default: all).",
    )
    parser.add_argument(
        "--pullback-indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Latent indices to include in the pullback metric.",
    )
    return parser.parse_args()


def _device(name: Optional[str]) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = _device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    decoder, template = decoder_from_checkpoint(checkpoint, device)
    baseline_latent = baseline_latent_vector(checkpoint).to(device)
    baseline_mesh = decode_latent(decoder, template, baseline_latent, device)
    base_verts, base_faces = (
        baseline_mesh.verts_packed().cpu().numpy(),
        baseline_mesh.faces_packed().cpu().numpy(),
    )

    latent_path = resolve_latent_path(
        args.latent_path, args.latent_root, args.latent_status, args.latent_index
    )
    print(f"[INFO] Using latent tensor: {latent_path}")
    latent = load_latent_tensor(latent_path).to(device)
    target_mesh = decode_latent(decoder, template, latent, device)
    verts, faces = (
        target_mesh.verts_packed().cpu().numpy(),
        target_mesh.faces_packed().cpu().numpy(),
    )

    deformation = compute_deformation_metrics(baseline_mesh, target_mesh)
    disp_norm = deformation.displacement_norm
    print(
        f"[RESULT] Displacement: mean={disp_norm.mean():.4e}, "
        f"max={disp_norm.max():.4e}"
    )
    print(
        f"[RESULT] Area change stats: "
        f"mean={np.mean(deformation.face_metrics['area_change']):.4f}, "
        f"max={np.max(deformation.face_metrics['area_change']):.4f}"
    )
    print(
        f"[RESULT] Stretch κ stats: "
        f"mean={np.mean(deformation.face_metrics['kappa']):.4f}, "
        f"max={np.max(deformation.face_metrics['kappa']):.4f}"
    )

    spectral = spectral_metrics(
        verts,
        faces,
        base_verts,
        base_faces,
        k=args.spectral_k,
        weights=args.spectral_weights,
        times=args.spectral_times,
    )
    print(
        f"[RESULT] Spectral distance (k={args.spectral_k}): "
        f"{spectral.spectrum_distance:.4e}"
    )
    print(
        f"[RESULT] HKS difference per-vertex: "
        f"mean={spectral.hks_distance_per_vertex.mean():.4e}, "
        f"max={spectral.hks_distance_per_vertex.max():.4e}"
    )

    fd = finite_difference_latent_sensitivity(
        decoder,
        template,
        baseline_latent,
        latent_indices=args.fd_indices,
        epsilon=args.fd_epsilon,
        device=device,
    )
    top_global = np.argsort(fd.global_scores)[::-1][:5]
    print("[RESULT] Top latent dimensions by global impact:")
    for rank, idx in enumerate(top_global, 1):
        print(f"  #{rank}: latent {idx} -> score {fd.global_scores[idx]:.4e}")

    metric, metric_indices = pullback_metric(
        decoder,
        template,
        baseline_latent,
        area_weights=None,
        device=device,
        latent_indices=args.pullback_indices,
        epsilon=args.fd_epsilon,
    )
    eigvals = np.linalg.eigvalsh(metric)
    print(
        f"[RESULT] Pullback metric eigenvalues (ordered for indices {metric_indices}): "
        f"{eigvals[::-1]}"
    )


if __name__ == "__main__":
    main()


