#!/usr/bin/env python3
"""Decode and evaluate a single latent vector from a MeshDecoder checkpoint."""

import argparse
import csv
import os
import sys
import time
from typing import Dict

import torch
import trimesh

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

    start = time.time()
    pred_mesh = decode_mesh(decoder, template, latent_vector)
    elapsed = time.time() - start

    os.makedirs(args.output_dir, exist_ok=True)
    mesh_path = os.path.join(args.output_dir, args.mesh_name)
    save_mesh(pred_mesh, mesh_path)
    print(f"Decoded mesh saved to {mesh_path}")

    if args.show:
        display_mesh(pred_mesh)

    target_mesh_path = args.target_mesh or resolve_default_target(
        checkpoint, args.latent_index
    )
    print(f"Using target mesh: {target_mesh_path}")

    if target_mesh_path:
        target_mesh = load_objs_as_meshes([target_mesh_path], device=device)
        metrics = compute_metrics(pred_mesh, target_mesh, args.metric_samples)
        summary = summarise_metrics(metrics, elapsed)
        metrics_path = os.path.join(args.output_dir, "metrics.csv")
        write_metrics_csv(metrics_path, summary)
        print(f"Metrics written to {metrics_path}")
        for key in sorted(summary.keys()):
            print(f"{key}: {summary[key]:.4f}")
    else:
        print("Target mesh could not be determined; metrics were not computed.")


if __name__ == "__main__":
    main()

