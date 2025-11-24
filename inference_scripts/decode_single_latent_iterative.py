#!/usr/bin/env python3
"""Decode and evaluate a single latent vector from a MeshDecoder checkpoint."""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import trimesh
import numpy as np

# Ensure project imports resolve when script is executed directly
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes

from model.mesh_decoder import MeshDecoder
from model.loss import mesh_bl_quality_loss
from util.metrics import point_metrics, self_intersections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode a single latent vector and compute per-mesh metrics."
    )
    parser.add_argument("--checkpoint", required=False, default = '/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_12-00-26.ckpt', help="Path to .ckpt file")
    parser.add_argument(
        "--latent-index",
        type=int,
        default=0,
        help="Index of latent vector to decode (default: 0)",
    )
    parser.add_argument(
        "--target-mesh",
        type=str,
        default=None,
        help="Optional path to reference mesh. Defaults to training mesh matching the latent index.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="decoded_mesh",
        help="Directory to store decoded mesh and metrics",
    )
    parser.add_argument(
        "--mesh-name",
        type=str,
        default="decoded_mesh.obj",
        help="Filename for the saved decoded mesh",
    )
    parser.add_argument(
        "--metric-samples",
        type=int,
        default=10_000,
        help="Number of points sampled per mesh for metric computation",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the decoded mesh in an interactive window",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of decode/analyse iterations before saving the mesh.",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=10.0,
        help="Percentage (0-100] of largest triangles to report each iteration.",
    )
    parser.add_argument(
        "--record-triangle-info",
        action="store_true",
        help="Write per-iteration JSON summaries of the selected top triangles.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        default=True,
        help="Skip metric computation against the target mesh (default: enabled).",
    )
    return parser.parse_args()


def to_device_mesh(mesh: Meshes, device: torch.device) -> Meshes:
    return mesh.to(device)


def instantiate_decoder(checkpoint: Dict, device: torch.device) -> MeshDecoder:
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
    return decoder


def extract_latent(latent_module, index: int, device: torch.device) -> torch.Tensor:
    if isinstance(latent_module, torch.nn.Embedding):
        vectors = latent_module.weight.detach()
    elif isinstance(latent_module, torch.nn.Parameter):
        vectors = latent_module.detach()
    else:
        vectors = torch.as_tensor(latent_module)
    if index < 0 or index >= vectors.shape[0]:
        raise IndexError(f"latent index {index} out of range (0-{vectors.shape[0]-1})")
    return vectors[index].unsqueeze(0).to(device)


def decode_mesh(
    decoder: MeshDecoder,
    template: Meshes,
    latent_vector: torch.Tensor,
) -> Meshes:
    with torch.no_grad():
        outputs = decoder(template.clone(), latent_vector)
    return outputs[-1]


def compute_face_areas(mesh: Meshes) -> torch.Tensor:
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    if faces.numel() == 0:
        return torch.zeros(0, device=verts.device)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0)
    return 0.5 * torch.linalg.norm(cross, dim=1)


def select_top_faces(mesh: Meshes, percent: float) -> Dict[str, torch.Tensor]:
    areas = compute_face_areas(mesh)
    num_faces = areas.shape[0]
    if num_faces == 0 or percent <= 0.0:
        return {
            "indices": torch.empty(0, dtype=torch.long, device=areas.device),
            "areas": torch.empty(0, device=areas.device),
            "threshold": 0.0,
        }
    percent = min(percent, 100.0)
    k = max(1, int(math.ceil(num_faces * (percent / 100.0))))
    k = min(k, num_faces)
    topk = torch.topk(areas, k, largest=True, sorted=True)
    threshold = topk.values[-1].item() if topk.values.numel() else 0.0
    return {"indices": topk.indices, "areas": topk.values, "threshold": threshold}


def record_top_face_summary(
    output_dir: str,
    iteration: int,
    percent: float,
    decoded_top: Dict[str, torch.Tensor],
    template: Meshes,
) -> None:
    indices = decoded_top["indices"]
    areas = decoded_top["areas"]
    if indices.numel() == 0:
        payload = {
            "iteration": iteration,
            "percent": percent,
            "face_indices": [],
            "decoded_face_areas": [],
            "template_faces": [],
            "threshold_area": 0.0,
        }
    else:
        template_faces = template.faces_packed()[indices]
        payload = {
            "iteration": iteration,
            "percent": percent,
            "face_indices": indices.detach().cpu().tolist(),
            "decoded_face_areas": areas.detach().cpu().tolist(),
            "template_faces": template_faces.detach().cpu().tolist(),
            "threshold_area": float(decoded_top["threshold"]),
        }
    out_path = Path(output_dir) / f"top_triangles_iter{iteration}.json"
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def refine_template(template: Meshes, face_indices: torch.Tensor) -> Meshes:
    if face_indices.numel() == 0:
        return template
    device = template.verts_packed().device
    dtype = template.verts_packed().dtype

    verts_np = template.verts_list()[0].detach().cpu().numpy()
    faces_np = template.faces_list()[0].detach().cpu().numpy()

    selected = {int(i) for i in face_indices.detach().cpu().tolist()}
    if not selected:
        return template

    edge_to_faces = {}
    face_edges = []
    for face_idx, face in enumerate(faces_np):
        v0, v1, v2 = map(int, face)
        edges = [
            tuple(sorted((v0, v1))),
            tuple(sorted((v1, v2))),
            tuple(sorted((v2, v0))),
        ]
        face_edges.append(edges)
        for edge in edges:
            edge_to_faces.setdefault(edge, []).append(face_idx)

    refine_set = set(selected)
    queue = list(selected)
    while queue:
        f_idx = queue.pop()
        for edge in face_edges[f_idx]:
            for neighbor in edge_to_faces.get(edge, []):
                if neighbor not in refine_set:
                    refine_set.add(neighbor)
                    queue.append(neighbor)

    verts_list = verts_np.tolist()
    result_faces = []
    edge_midpoints = {}

    def midpoint_index(i: int, j: int) -> int:
        key = tuple(sorted((i, j)))
        existing = edge_midpoints.get(key)
        if existing is not None:
            return existing
        midpoint = (np.array(verts_list[i]) + np.array(verts_list[j])) * 0.5
        verts_list.append(midpoint.tolist())
        new_idx = len(verts_list) - 1
        edge_midpoints[key] = new_idx
        return new_idx

    for face_idx, face in enumerate(faces_np):
        v0, v1, v2 = map(int, face)
        if face_idx not in refine_set:
            result_faces.append([v0, v1, v2])
            continue
        m01 = midpoint_index(v0, v1)
        m12 = midpoint_index(v1, v2)
        m20 = midpoint_index(v2, v0)
        result_faces.extend(
            [
                [v0, m01, m20],
                [v1, m12, m01],
                [v2, m20, m12],
                [m01, m12, m20],
            ]
        )

    new_verts = torch.tensor(verts_list, dtype=dtype, device=device)
    new_faces = torch.tensor(result_faces, dtype=torch.long, device=device)
    return Meshes(verts=[new_verts], faces=[new_faces])


def save_mesh(mesh: Meshes, output_path: str) -> None:
    mesh_cpu = mesh.cpu()
    vertices = mesh_cpu.verts_list()[0].numpy()
    faces = mesh_cpu.faces_list()[0].numpy()
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    directory = os.path.dirname(output_path) or "."
    os.makedirs(directory, exist_ok=True)
    tri_mesh.export(output_path)


def display_mesh(mesh: Meshes) -> None:
    mesh_cpu = mesh.cpu()
    tri_mesh = trimesh.Trimesh(
        vertices=mesh_cpu.verts_list()[0].numpy(),
        faces=mesh_cpu.faces_list()[0].numpy(),
        process=False,
    )
    tri_mesh.show()


def compute_metrics(
    pred_mesh: Meshes,
    target_mesh: Meshes,
    num_samples: int,
) -> Dict[str, float]:
    cf = 10000
    with torch.no_grad():
        pred_samples = sample_points_from_meshes(pred_mesh, num_samples)
        true_samples = sample_points_from_meshes(target_mesh, num_samples)
        chamfer_val = chamfer_distance(true_samples, pred_samples)[0] * cf
        metrics = point_metrics(true_samples, pred_samples, [0.01, 0.02])
        bl_quality = (1.0 - mesh_bl_quality_loss(pred_mesh)).item()

    pred_mesh_cpu = pred_mesh.cpu()
    ints_tensor, _ = self_intersections(pred_mesh_cpu)
    faces_count = len(pred_mesh_cpu.faces_packed())
    ints_percent = 100.0 * float(ints_tensor[0]) / max(faces_count, 1)

    out = {
        "ChamferL2 x 10000": chamfer_val.item(),
        "BL quality": bl_quality,
        "No. ints.": ints_percent,
        "Precision@0.01": metrics["Precision@0.01"].item(),
        "Recall@0.01": metrics["Recall@0.01"].item(),
        "F1@0.01": metrics["F1@0.01"].item(),
        "Precision@0.02": metrics["Precision@0.02"].item(),
        "Recall@0.02": metrics["Recall@0.02"].item(),
        "F1@0.02": metrics["F1@0.02"].item(),
    }

    return out


def summarise_metrics(raw_metrics: Dict[str, float], elapsed: float) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key, value in raw_metrics.items():
        summary[f"{key}_mean"] = float(value)
        summary[f"{key}_std"] = 0.0
    summary["Search_mean"] = elapsed
    summary["Search_std"] = 0.0
    summary["Total_mean"] = elapsed
    summary["Total_std"] = 0.0
    summary["num_test_samples"] = 1
    return summary


def latent_count(latent_module) -> int:
    if isinstance(latent_module, torch.nn.Embedding):
        return latent_module.weight.shape[0]
    if isinstance(latent_module, torch.nn.Parameter):
        return latent_module.shape[0]
    latent_tensor = torch.as_tensor(latent_module)
    return latent_tensor.shape[0]


def resolve_default_target(checkpoint: Dict, index: int) -> str:
    train_path = checkpoint.get("train_data_path")
    filenames = checkpoint.get("train_filenames")
    if not train_path or not filenames:
        raise ValueError(
            "Checkpoint does not contain training mesh metadata; please pass --target-mesh explicitly."
        )
    if index < 0 or index >= len(filenames):
        raise IndexError(
            f"latent index {index} out of range for available training meshes (0-{len(filenames)-1})"
        )
    return os.path.join(train_path, filenames[index])


def write_metrics_csv(path: str, metrics: Dict[str, float]) -> None:
    columns = [
        "ChamferL2 x 10000_mean",
        "ChamferL2 x 10000_std",
        "BL quality_mean",
        "BL quality_std",
        "No. ints._mean",
        "No. ints._std",
        "Precision@0.01_mean",
        "Precision@0.01_std",
        "Recall@0.01_mean",
        "Recall@0.01_std",
        "F1@0.01_mean",
        "F1@0.01_std",
        "Precision@0.02_mean",
        "Precision@0.02_std",
        "Recall@0.02_mean",
        "Recall@0.02_std",
        "F1@0.02_mean",
        "F1@0.02_std",
        "Search_mean",
        "Search_std",
        "Total_mean",
        "Total_std",
        "num_test_samples",
    ]
    row = {column: metrics.get(column, "") for column in columns}
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")
    if not (0.0 < args.top_percent <= 100.0):
        raise ValueError("--top-percent must be within (0, 100].")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    decoder = instantiate_decoder(checkpoint, device)
    template = to_device_mesh(checkpoint["template"], device)
    latent_module = checkpoint["latent_vectors"]
    total_latents = latent_count(latent_module)
    print(f"Loaded checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"Device: {device}")
    print(f"Latent vectors available: {total_latents}")
    latent_vector = extract_latent(latent_module, args.latent_index, device)
    print(f"Decoding latent index {args.latent_index}")

    os.makedirs(args.output_dir, exist_ok=True)

    final_mesh: Meshes = None  # type: ignore
    decode_elapsed: float = 0.0
    iterations = args.iterations
    top_percent = args.top_percent

    for iteration in range(1, iterations + 1):
        print(f"\n[Iteration {iteration}/{iterations}] Starting decode.")
        start = time.time()
        pred_mesh = decode_mesh(decoder, template, latent_vector)
        decode_elapsed = time.time() - start
        final_mesh = pred_mesh
        print(
            f"[Iteration {iteration}] Decode complete in {decode_elapsed:.3f}s "
            f"({pred_mesh.faces_packed().shape[0]} faces)."
        )

        top_info = select_top_faces(pred_mesh, top_percent)
        top_count = int(top_info["indices"].numel())
        if top_count == 0:
            print(f"[Iteration {iteration}] Mesh contains no faces to analyse.")
        else:
            threshold = top_info["threshold"]
            print(
                f"[Iteration {iteration}] Top {top_percent:.2f}% corresponds to "
                f"{top_count} faces; smallest area in subset {threshold:.6f}."
            )
            preview_count = min(10, top_count)
            preview_indices = (
                top_info["indices"][:preview_count].detach().cpu().tolist()
            )
            print(
                f"[Iteration {iteration}] Example template face indices: "
                f"{preview_indices}"
            )
            if args.record_triangle_info:
                record_top_face_summary(
                    args.output_dir, iteration, top_percent, top_info, template
                )

        if top_count > 0 and iteration < iterations:
            before_faces = template.faces_packed().shape[0]
            template = refine_template(template, top_info["indices"])
            after_faces = template.faces_packed().shape[0]
            print(
                f"[Iteration {iteration}] Template refined: "
                f"{before_faces} -> {after_faces} faces."
            )

    if final_mesh is None:
        raise RuntimeError("Decoding failed to produce a mesh.")

    mesh_name = Path(args.mesh_name)
    suffix = mesh_name.suffix or ".obj"
    final_mesh_name = f"{mesh_name.stem}_iter{iterations}{suffix}"
    mesh_path = os.path.join(args.output_dir, final_mesh_name)
    save_mesh(final_mesh, mesh_path)
    print(f"Decoded mesh saved to {mesh_path}")

    if args.show:
        display_mesh(final_mesh)

    if args.no_metrics:
        print("Metric computation skipped (--no-metrics).")
    else:
        target_mesh_path = args.target_mesh or resolve_default_target(
            checkpoint, args.latent_index
        )
        print(f"Using target mesh: {target_mesh_path}")

        if target_mesh_path:
            target_mesh = load_objs_as_meshes([target_mesh_path], device=device)
            metrics = compute_metrics(final_mesh, target_mesh, args.metric_samples)
            summary = summarise_metrics(metrics, decode_elapsed)
            metrics_filename = f"metrics_iter{iterations}.csv"
            metrics_path = os.path.join(args.output_dir, metrics_filename)
            write_metrics_csv(metrics_path, summary)
            print(f"Metrics written to {metrics_path}")
            for key in sorted(summary.keys()):
                print(f"{key}: {summary[key]:.4f}")
        else:
            print("Target mesh could not be determined; metrics were not computed.")


if __name__ == "__main__":
    main()

