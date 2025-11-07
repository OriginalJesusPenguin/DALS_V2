CHECKPOINT_DEFAULT = "/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_12-00-26.ckpt"
LATENT_DIR_DEFAULT = "/home/ralbe/DALS/mesh_autodecoder/inference_results/meshes_MeshDecoderTrainer_2025-11-06_12-00-26/latents"
TARGET_DIR_DEFAULT = "/home/ralbe/DALS/mesh_autodecoder/inference_results/meshes_MeshDecoderTrainer_2025-11-06_12-00-26"
#!/usr/bin/env python3
"""Mask latent vectors, decode meshes, and evaluate robustness metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

# Ensure project modules resolve when executed as a script
PROJECT_ROOT = "/home/ralbe/DALS/mesh_autodecoder"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes

from model.mesh_decoder import MeshDecoder
from model.loss import mesh_bl_quality_loss
from util.metrics import point_metrics, self_intersections


@dataclass
class TargetInfo:
    name: str
    status: str
    patient_id: Optional[str]
    target_path: str
    samples: torch.Tensor
    samples_np: np.ndarray
    lb_vertices: np.ndarray
    lb_faces: np.ndarray
    lb_eigs: np.ndarray


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint["hparams"]
    decoder = MeshDecoder(
        hparams["latent_features"],
        hparams["steps"],
        hparams["hidden_features"],
        hparams["subdivide"],
        mode=hparams["decoder_mode"],
        norm=hparams["normalization"][0],
    )
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    decoder.eval()
    decoder.to(device)

    template: Meshes = checkpoint["template"].to(device)

    return {
        "checkpoint": checkpoint,
        "decoder": decoder,
        "template": template,
    }


def resolve_status_and_id(name: str) -> Dict[str, Optional[str]]:
    lowered = name.lower()
    status = "cirrhotic" if "cirrhotic" in lowered else "healthy" if "healthy" in lowered else "unknown"
    match = re.search(r"(\d+)", name)
    patient_id = match.group(1) if match else None
    return {"status": status, "patient_id": patient_id}


def discover_latents(latent_dir: str) -> List[str]:
    entries = sorted(
        [f for f in os.listdir(latent_dir) if f.endswith(".pt")],
        key=lambda x: x,
    )
    if not entries:
        raise FileNotFoundError(f"No latent vectors found in {latent_dir}")
    return entries


def load_latent(path: str, device: torch.device) -> torch.Tensor:
    latent = torch.load(path, map_location="cpu")
    if isinstance(latent, torch.Tensor):
        tensor = latent.detach().cpu()
    else:
        tensor = torch.as_tensor(latent)
    tensor = tensor.view(-1)
    return tensor.to(device)


def load_target_info(
    name: str,
    latent_dir: str,
    target_dir: Optional[str],
    device: torch.device,
    metric_samples: int,
    emd_samples: int,
    lb_k: int,
) -> TargetInfo:
    parsed = resolve_status_and_id(name)
    status = parsed["status"]
    patient_id = parsed["patient_id"]

    if target_dir:
        candidate = os.path.join(target_dir, f"{status}_{patient_id}_testing_target.obj") if patient_id else None
    else:
        base = os.path.dirname(latent_dir.rstrip(os.sep))
        candidate = os.path.join(base, f"{status}_{patient_id}_testing_target.obj") if patient_id else None

    if not candidate or not os.path.exists(candidate):
        raise FileNotFoundError(f"Target mesh not found for latent {name} (expected {candidate})")

    mesh_cpu = load_objs_as_meshes([candidate], device=torch.device("cpu"))
    verts_np = mesh_cpu.verts_list()[0].numpy()
    faces_np = mesh_cpu.faces_list()[0].numpy()

    mesh_device = mesh_cpu.to(device)
    samples = sample_points_from_meshes(mesh_device, metric_samples)[0].detach().cpu()
    samples_np = samples[: min(emd_samples, samples.shape[0])].numpy()

    lb_eigs = laplace_beltrami_eigs(verts_np, faces_np, k=lb_k)

    return TargetInfo(
        name=name,
        status=status,
        patient_id=patient_id,
        target_path=candidate,
        samples=samples,
        samples_np=samples_np,
        lb_vertices=verts_np,
        lb_faces=faces_np,
        lb_eigs=lb_eigs,
    )


def generate_masks(
    latent_dim: int,
    num_masks: int,
    prob: float,
    device: torch.device,
) -> torch.Tensor:
    masks = torch.bernoulli(
        torch.full((num_masks, latent_dim), prob, device=device)
    )
    return masks


def sample_pred_points(pred_mesh: Meshes, num_samples: int) -> torch.Tensor:
    return sample_points_from_meshes(pred_mesh, num_samples)[0]


def compute_chamfer_metrics(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    precision_thresholds: Iterable[float],
) -> Dict[str, float]:
    if pred_points.dim() != 3 or target_points.dim() != 3:
        raise ValueError("Chamfer metric expects batched point clouds of shape (B, N, 3)")

    chamfer_val = chamfer_distance(target_points, pred_points)[0]
    result: Dict[str, float] = {
        "ChamferL2_x_10000": float(chamfer_val.item() * 10000.0)
    }

    metrics = point_metrics(target_points, pred_points, precision_thresholds)
    for key, value in metrics.items():
        result[key] = float(value.item())

    return result


def compute_mesh_metrics(pred_mesh: Meshes) -> Dict[str, float]:
    with torch.no_grad():
        bl_quality = float((1.0 - mesh_bl_quality_loss(pred_mesh)).item())
    pred_mesh_cpu = pred_mesh.cpu()
    ints_tensor, _ = self_intersections(pred_mesh_cpu)
    faces_count = len(pred_mesh_cpu.faces_packed())
    ints_percent = 100.0 * float(ints_tensor[0]) / max(faces_count, 1)
    return {
        "BL_quality": bl_quality,
        "No_ints_percent": ints_percent,
    }


def pairwise_distances(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    P2 = np.sum(P * P, axis=1)[:, None]
    Q2 = np.sum(Q * Q, axis=1)[None, :]
    C2 = P2 + Q2 - 2.0 * (P @ Q.T)
    np.maximum(C2, 0.0, out=C2)
    return np.sqrt(C2)


def emd_via_linprog(P: np.ndarray, Q: np.ndarray) -> float:
    n, m = P.shape[0], Q.shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    C = pairwise_distances(P, Q)
    c_vec = C.flatten()

    rows = n + m
    cols = n * m
    A_eq = sparse.lil_matrix((rows, cols), dtype=np.float64)
    b_eq = np.zeros(rows, dtype=np.float64)

    for i in range(n):
        start = i * m
        A_eq[i, start : start + m] = 1.0
        b_eq[i] = a[i]

    for j in range(m):
        A_eq[n + j, j : cols : m] = 1.0
        b_eq[n + j] = b[j]

    A_eq = A_eq.tocsr()
    bounds = [(0, None)] * cols
    res = linprog(c_vec, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Linear program failed: {res.message}")
    return float(res.fun)


def compute_emd(P: np.ndarray, Q: np.ndarray) -> float:
    try:
        import ot

        a = np.full(P.shape[0], 1.0 / P.shape[0])
        b = np.full(Q.shape[0], 1.0 / Q.shape[0])
        C = pairwise_distances(P, Q)
        return float(ot.emd2(a, b, C))
    except Exception:
        return emd_via_linprog(P, Q)


def triangle_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    v0 = V[F[:, 0], :]
    v1 = V[F[:, 1], :]
    v2 = V[F[:, 2], :]
    cross_prod = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross_prod, axis=1)


def cotangent(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = b - a
    v = c - a
    cross_n = np.cross(u, v)
    denom = np.linalg.norm(cross_n)
    if denom < 1e-16:
        return 0.0
    return float(np.dot(u, v) / denom)


def build_cotangent_laplacian(V: np.ndarray, F: np.ndarray) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    n_verts = V.shape[0]
    I: List[int] = []
    J: List[int] = []
    W: List[float] = []
    areas = triangle_areas(V, F)
    vertex_area = np.zeros(n_verts, dtype=np.float64)

    for idx, (i, j, k) in enumerate(F):
        vi, vj, vk = V[i], V[j], V[k]
        a = areas[idx]
        vertex_area[i] += a / 3.0
        vertex_area[j] += a / 3.0
        vertex_area[k] += a / 3.0

        cot_gamma = cotangent(vk, vi, vj)
        cot_alpha = cotangent(vj, vi, vk)
        cot_beta = cotangent(vi, vj, vk)

        w_ij = 0.5 * cot_gamma
        w_jk = 0.5 * cot_alpha
        w_ki = 0.5 * cot_beta

        I += [i, j, j, k, k, i]
        J += [j, i, k, j, i, k]
        W += [w_ij, w_ij, w_jk, w_jk, w_ki, w_ki]

    Wmat = sparse.coo_matrix((W, (I, J)), shape=(n_verts, n_verts), dtype=np.float64).tocsr()
    diag = np.array(Wmat.sum(axis=1)).ravel()
    L = sparse.diags(diag, 0) - Wmat
    M = sparse.diags(vertex_area, 0)
    return L, M


def laplace_beltrami_eigs(V: np.ndarray, F: np.ndarray, k: int) -> np.ndarray:
    if V.shape[0] == 0 or F.shape[0] == 0 or k <= 0:
        return np.zeros(max(k, 1), dtype=np.float64)[:k]

    L, M = build_cotangent_laplacian(V, F)
    max_k = min(k, max(1, L.shape[0] - 1))
    if max_k == 0:
        return np.zeros(k, dtype=np.float64)

    reg = 1e-12
    evals, _ = eigsh(L + reg * M, k=max_k, M=M, sigma=0.0, which="LM")
    evals = np.real(np.sort(evals))
    if evals.size < k:
        padded = np.zeros(k, dtype=np.float64)
        padded[: evals.size] = evals
        return padded
    return evals[:k]


def decode_mesh(decoder: MeshDecoder, template: Meshes, latent: torch.Tensor) -> Meshes:
    with torch.no_grad():
        outputs = decoder(template.clone(), latent.unsqueeze(0))
    return outputs[-1]


def format_list(values: np.ndarray) -> str:
    return json.dumps([float(v) for v in values.tolist()])


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_pipeline(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    ensure_directory(args.output_dir)

    ckpt_bundle = load_checkpoint(args.checkpoint, device)
    decoder: MeshDecoder = ckpt_bundle["decoder"]
    template: Meshes = ckpt_bundle["template"]

    latent_files = discover_latents(args.latent_dir)
    targets: Dict[str, TargetInfo] = {}
    latents: Dict[str, torch.Tensor] = {}

    for filename in tqdm(latent_files, desc="Loading latents", leave=False):
        name = filename.replace("_latent.pt", "").replace(".pt", "")
        latent_path = os.path.join(args.latent_dir, filename)
        latent_tensor = load_latent(latent_path, device)
        latents[name] = latent_tensor
        targets[name] = load_target_info(
            name,
            args.latent_dir,
            args.target_dir,
            device,
            args.metric_samples,
            args.emd_samples,
            args.lb_k,
        )

    latent_dim = next(iter(latents.values())).numel()
    masks = generate_masks(latent_dim, args.num_masks, args.mask_prob, device)

    records: List[Dict[str, object]] = []

    for mask_idx, mask in enumerate(tqdm(masks, desc="Applying masks")):
        mask_ratio = float(mask.mean().item())
        for name, latent_tensor in latents.items():
            target = targets[name]
            masked_latent = latent_tensor * mask

            pred_mesh = decode_mesh(decoder, template, masked_latent)
            pred_points = sample_pred_points(pred_mesh, args.metric_samples).detach()
            target_points = target.samples
            if target_points.device != pred_points.device:
                target_points = target_points.to(pred_points.device)
            pred_batch = pred_points.unsqueeze(0)
            target_batch = target_points.unsqueeze(0)
            metrics = compute_chamfer_metrics(pred_batch, target_batch, [0.01, 0.02])
            mesh_metrics = compute_mesh_metrics(pred_mesh)

            pred_points_np = pred_points[: min(args.emd_samples, pred_points.shape[0])].detach().cpu().numpy()
            emd_value = compute_emd(pred_points_np, target.samples_np)

            verts_np = pred_mesh.verts_list()[0].detach().cpu().numpy()
            faces_np = pred_mesh.faces_list()[0].detach().cpu().numpy()
            lb_pred = laplace_beltrami_eigs(verts_np, faces_np, args.lb_k)

            record = {
                "checkpoint": os.path.basename(args.checkpoint),
                "latent_name": name,
                "mask_index": mask_idx,
                "mask_keep_ratio": mask_ratio,
                "status": target.status,
                "patient_id": target.patient_id,
                "target_path": target.target_path,
                "emd": emd_value,
                "lb_pred_eigs": format_list(lb_pred),
                "lb_target_eigs": format_list(target.lb_eigs),
                "lb_l2_delta": float(np.linalg.norm(lb_pred - target.lb_eigs)),
            }
            record.update(metrics)
            record.update(mesh_metrics)
            records.append(record)

            if args.flush_cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame.from_records(records)
    output_base = os.path.join(args.output_dir, "relax_metrics")
    output_path = output_base + ".parquet"
    output_format = "parquet"
    try:
        df.to_parquet(output_path, index=False)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        output_path = output_base + ".csv"
        output_format = "csv"
        df.to_csv(output_path, index=False)
        print(f"Parquet export unavailable ({exc}); wrote CSV instead at {output_path}")

    metadata = {
        "checkpoint": args.checkpoint,
        "latent_dir": args.latent_dir,
        "target_dir": args.target_dir,
        "num_masks": args.num_masks,
        "mask_prob": args.mask_prob,
        "metric_samples": args.metric_samples,
        "emd_samples": args.emd_samples,
        "lb_k": args.lb_k,
        "device": str(device),
        "records": len(records),
        "output_path": output_path,
        "output_format": output_format,
    }
    with open(os.path.join(args.output_dir, "relax_metrics_meta.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved metrics to {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate masked latent robustness metrics")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT_DEFAULT,
        help="Path to MeshDecoder checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--latent-dir",
        type=str,
        default=LATENT_DIR_DEFAULT,
        help="Directory containing *_latent.pt files",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=TARGET_DIR_DEFAULT,
        help="Directory containing target meshes. Defaults to parent of latent dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ralbe/DALS/mesh_autodecoder/relax_explanations",
        help="Directory to store metrics outputs",
    )
    parser.add_argument("--num-masks", type=int, default=10, help="Number of random masks")
    parser.add_argument("--mask-prob", type=float, default=0.5, help="Bernoulli keep probability")
    parser.add_argument("--metric-samples", type=int, default=10000, help="Points sampled per mesh for metrics")
    parser.add_argument("--emd-samples", type=int, default=512, help="Points per mesh for EMD computation")
    parser.add_argument("--lb-k", type=int, default=10, help="Number of Laplace-Beltrami eigenvalues")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--flush-cuda", action="store_true", help="Clear CUDA cache after each decode")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()


