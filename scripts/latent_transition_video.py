#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pytorch3d.structures import Meshes
# import pytorch3d.subdivide_meshes
from pytorch3d.ops import subdivide_meshes

import imageio.v2 as imageio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.mesh_decoder import MeshDecoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_checkpoint = PROJECT_ROOT / "models" / "MeshDecoderTrainer_2025-11-06_12-00-26.ckpt"
    parser = argparse.ArgumentParser(
        description="Render a latent transition as frame sequence and video."
    )
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--latent-path", type=Path, required=False,default = '/home/ralbe/DALS/mesh_autodecoder/inference_results/meshes_MeshDecoderTrainer_2025-11-06_12-00-26/latents/cirrhotic_115_testing_latent.pt')
    parser.add_argument("--latent-index", type=int, required=False, default=66)
    parser.add_argument("--min-value", type=float, required=True)
    parser.add_argument("--max-value", type=float, required=True)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "video_frames")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def ensure_output_dir(base: Path, latent_path: Path, latent_index: int) -> Path:
    stem = latent_path.stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = base / f"{stem}_idx{latent_index}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[MeshDecoder, Meshes, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams: Dict = checkpoint["hparams"]
    decoder = MeshDecoder(
        latent_features=hparams["latent_features"],
        steps=hparams["steps"],
        hidden_features=hparams["hidden_features"],
        subdivide=hparams["subdivide"],
        mode=hparams["decoder_mode"],
        norm=hparams["normalization"][0],
    )
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    decoder.to(device)
    decoder.eval()
    template: Meshes = checkpoint["template"].to(device)
    subdiv = subdivide_meshes.SubdivideMeshes()
    template = subdiv(template).to(device)
    return decoder, template, hparams


def load_latent_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, dict):
        for key in ("latent", "latent_vector", "latent_vectors"):
            if key in tensor:
                tensor = tensor[key]
                break
        else:
            raise KeyError(f"No latent vector found in {path}")
    tensor = torch.as_tensor(tensor).float()
    if tensor.ndim == 2:
        tensor = tensor.squeeze(0)
    return tensor.clone()


def decode_latent(decoder: MeshDecoder, template: Meshes, latent: torch.Tensor, device: torch.device) -> Meshes:
    latent_batch = latent.unsqueeze(0).to(device)
    with torch.no_grad():
        decoded = decoder(template.clone(), latent_batch)[-1]
    return decoded.to("cpu")


def mesh_bounds(mesh: Meshes) -> Tuple[np.ndarray, np.ndarray]:
    verts = mesh.verts_packed().cpu().numpy()
    return verts.min(axis=0), verts.max(axis=0)


def mesh_to_wireframe_trace(
    mesh: Meshes,
    color: str,
    width: int,
    name: str,
    center: np.ndarray,
) -> go.Scatter3d:
    verts = mesh.verts_packed().cpu().numpy() - center
    edges = mesh.edges_packed().cpu().numpy()
    coords = np.empty((3 * len(edges), 3), dtype=np.float32)
    coords[0::3] = verts[edges[:, 0]]
    coords[1::3] = verts[edges[:, 1]]
    coords[2::3] = np.nan
    return go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="lines",
        name=name,
        line=dict(color=color, width=width),
        hoverinfo="skip",
    )


def axis_ranges(mins: np.ndarray, maxs: np.ndarray) -> List[List[float]]:
    center = 0.5 * (mins + maxs)
    span = max(maxs - mins)
    half = max(span / 2.0, 1e-3)
    return [[center[i] - half, center[i] + half] for i in range(3)]


def orbit_camera(angle: float, radius: float) -> Dict[str, Dict[str, float]]:
    return {
        "eye": {
            "x": radius * np.cos(angle),
            "y": radius * np.sin(angle),
            "z": 0.35 * radius,
        },
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    }


def figure_for_mesh(
    original: Meshes,
    current: Meshes,
    ranges: List[List[float]],
    step_value: float,
    camera: Dict[str, Dict[str, float]],
    center: np.ndarray,
    title_prefix: str,
) -> go.Figure:
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    fig.add_trace(mesh_to_wireframe_trace(original, "#636EFA", 1, "Original", center))
    fig.add_trace(mesh_to_wireframe_trace(current, "#EF553B", 3, "Modified", center))
    fig.update_layout(
        title=f"{title_prefix} • value → {step_value:+.4f}",
        scene=dict(
            xaxis=dict(visible=False, range=ranges[0]),
            yaxis=dict(visible=False, range=ranges[1]),
            zaxis=dict(visible=False, range=ranges[2]),
            aspectmode="cube",
            camera=camera,
        ),
        showlegend=False,
        height=1200,
        width=1600,
        margin=dict(l=0, r=0, t=70, b=0),
    )
    return fig


def accumulate_bounds(meshes: Iterable[Meshes]) -> Tuple[np.ndarray, np.ndarray]:
    mins, maxs = None, None
    for mesh in meshes:
        m_min, m_max = mesh_bounds(mesh)
        if mins is None:
            mins, maxs = m_min, m_max
        else:
            mins = np.minimum(mins, m_min)
            maxs = np.maximum(maxs, m_max)
    if mins is None or maxs is None:
        raise ValueError("No meshes were provided to compute bounds.")
    return mins, maxs


def write_video(frame_paths: List[Path], video_path: Path, fps: int) -> None:
    try:
        writer = imageio.get_writer(
            str(video_path),
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            macro_block_size=None,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "FFmpeg support is required to write mp4 videos. Install `imageio-ffmpeg` or ensure ffmpeg is on PATH."
        ) from exc

    with writer:
        for frame_path in frame_paths:
            frame = imageio.imread(frame_path)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = ensure_output_dir(output_root, args.latent_path, args.latent_index)
    decoder, template, _ = load_checkpoint(args.checkpoint, device)
    latent = load_latent_tensor(args.latent_path)

    if not 0 <= args.latent_index < latent.numel():
        raise IndexError(f"Latent index {args.latent_index} is out of bounds for tensor of size {latent.numel()}.")
    if args.steps < 2:
        raise ValueError("Steps must be at least 2 to form a transition.")

    original_value = float(latent[args.latent_index])
    original_mesh = decode_latent(decoder, template, latent, device)

    step_values = np.linspace(args.min_value, args.max_value, args.steps, dtype=np.float32)
    modified_meshes: List[Meshes] = []
    for value in step_values:
        modified = latent.clone()
        modified[args.latent_index] = float(value)
        mesh = decode_latent(decoder, template, modified, device)
        modified_meshes.append(mesh)

    bounds_min, bounds_max = accumulate_bounds([original_mesh, *modified_meshes])
    center = 0.5 * (bounds_min + bounds_max)
    ranges = axis_ranges(bounds_min - center, bounds_max - center)
    span = max(r[1] - r[0] for r in ranges)
    radius = max(span * 1.6, 1.0)

    title_prefix = f"{args.latent_path.stem} • latent[{args.latent_index}]"

    frame_paths: List[Path] = []
    for idx, (value, mesh) in enumerate(zip(step_values, modified_meshes)):
        angle = 2.0 * np.pi * idx / max(len(step_values) - 1, 1)
        camera = orbit_camera(angle, radius)
        fig = figure_for_mesh(
            original_mesh,
            mesh,
            ranges,
            float(value),
            camera,
            center,
            title_prefix,
        )
        fig.add_annotation(
            text=f"Original value: {original_value:+.4f}",
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.02,
            showarrow=False,
            font=dict(size=12),
        )
        frame_path = output_dir / f"frame_{idx:04d}.png"
        fig.write_image(str(frame_path), width=1600, height=1200, scale=1)
        frame_paths.append(frame_path)

    video_path = output_dir / "transition.mp4"
    try:
        write_video(frame_paths, video_path, args.fps)
    except Exception as exc:
        print(f"Could not create video at {video_path}: {exc}")
    else:
        print(f"Wrote video to {video_path}")

    if not args.keep_frames:
        for frame in frame_paths:
            frame.unlink(missing_ok=True)


if __name__ == "__main__":
    main()


