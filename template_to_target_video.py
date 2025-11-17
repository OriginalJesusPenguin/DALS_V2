#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pytorch3d.structures import Meshes
# import pytorch3d.subdivide_meshes
from pytorch3d.ops import subdivide_meshes
from tqdm.auto import tqdm

import imageio.v2 as imageio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.mesh_decoder import MeshDecoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_checkpoint = PROJECT_ROOT / "models" / "MeshDecoderTrainer_2025-11-06_12-00-26.ckpt"
    default_checkpoint = '/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_12-00-26.ckpt'
    default_latent_root = Path(
        "/home/ralbe/DALS/mesh_autodecoder/inference_results/meshes_MeshDecoderTrainer_2025-11-06_12-00-26/latents"
    )
    parser = argparse.ArgumentParser(
        description="Render the template-to-target morph as frames and video."
    )
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument(
        "--latent-path",
        type=Path,
        required=False,
        default=None,
        help="Optional explicit latent path; overrides status/index selection.",
    )
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=default_latent_root,
        help="Directory containing latent code files.",
    )
    parser.add_argument(
        "--latent-status",
        choices=("healthy", "cirrhotic"),
        default="cirrhotic",
        help="Latent cohort to sample from when --latent-path is not provided.",
    )
    parser.add_argument(
        "--latent-index",
        type=int,
        default=0,
        help="Zero-based index within the selected cohort (0-9).",
    )
    parser.add_argument("--steps", type=int, default=60, help="Number of frames from template to target.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "video_frames")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument(
        "--render-style",
        choices=("surface", "wireframe"),
        default="surface",
        help="Choose between a colored surface or wireframe visualization.",
    )
    parser.add_argument(
        "--rotate-camera",
        dest="rotate_camera",
        action="store_true",
        help="Orbit the camera throughout the morph.",
    )
    parser.add_argument(
        "--no-rotate-camera",
        dest="rotate_camera",
        action="store_false",
        help="Keep the camera fixed throughout the morph.",
    )
    parser.set_defaults(rotate_camera=True)
    return parser.parse_args()


def ensure_output_dir(base: Path, stem: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [
            path
            for path in base.iterdir()
            if path.is_dir() and path.name.startswith(f"{stem}_")
        ]
    )
    if existing:
        last = existing[-1].name
        try:
            suffix = int(last.split("_")[-1])
            next_idx = suffix + 1
        except (ValueError, IndexError):
            next_idx = len(existing)
    else:
        next_idx = 1
    out_dir = base / f"{stem}_{next_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=False)
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


def resolve_latent_path(
    latent_path: Path | None,
    latent_root: Path,
    status: str,
    index: int,
) -> Path:
    if latent_path is not None:
        return latent_path

    root = latent_root
    if not root.exists():
        raise FileNotFoundError(f"Latent root directory {root} does not exist.")

    candidates: List[Path] = []
    for pattern in (f"{status}_*.pt", f"{status}_*.pth"):
        candidates.extend(root.glob(pattern))
    candidates = sorted(candidates)

    if not candidates:
        raise FileNotFoundError(
            f"No latent codes found for status '{status}' in {root}."
        )
    if not 0 <= index < len(candidates):
        raise IndexError(
            f"Latent index {index} is out of bounds for status '{status}'. "
            f"Found {len(candidates)} candidates."
        )

    return candidates[index]


def decode_latent(decoder: MeshDecoder, template: Meshes, latent: torch.Tensor, device: torch.device) -> Meshes:
    latent_batch = latent.unsqueeze(0).to(device)
    with torch.no_grad():
        decoded = decoder(template.clone(), latent_batch)[-1]
    return decoded.to("cpu")


def mesh_bounds(mesh: Meshes) -> Tuple[np.ndarray, np.ndarray]:
    verts = mesh.verts_packed().cpu().numpy()
    return verts.min(axis=0), verts.max(axis=0)


def spherical_vertex_colors(verts: np.ndarray) -> List[str]:
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("Vertex array must have shape (N, 3).")
    if verts.shape[0] == 0:
        raise ValueError("Meshes must contain at least one vertex.")

    center = verts.mean(axis=0, keepdims=True)
    shifted = verts - center
    radii = np.linalg.norm(shifted, axis=1)
    radii = np.where(radii > 1e-12, radii, 1e-12)

    theta = (np.arctan2(shifted[:, 1], shifted[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
    phi = np.arccos(np.clip(shifted[:, 2] / radii, -1.0, 1.0))

    hue = theta / (2.0 * np.pi)
    saturation = 0.4 + 0.6 * (1.0 - phi / np.pi)
    value = 0.5 + 0.5 * np.cos(phi - (np.pi / 2.0))

    colors: List[str] = []
    for h, s, v in zip(hue, saturation, value):
        r, g, b = colorsys.hsv_to_rgb(h, np.clip(s, 0.0, 1.0), np.clip(v, 0.0, 1.0))
        colors.append(f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})")
    return colors


def interpolate_vertices(
    template_verts: np.ndarray, target_verts: np.ndarray, steps: int
) -> List[Tuple[float, np.ndarray]]:
    if template_verts.shape != target_verts.shape:
        raise ValueError("Template and target vertex arrays must share the same shape.")
    if steps < 2:
        raise ValueError("Steps must be at least 2 to form a transition.")
    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    template_np = np.asarray(template_verts, dtype=np.float32)
    target_np = np.asarray(target_verts, dtype=np.float32)
    return [
        (float(alpha), (1.0 - float(alpha)) * template_np + float(alpha) * target_np)
        for alpha in alphas
    ]


def axis_ranges(
    mins: np.ndarray, maxs: np.ndarray, padding: float = 0.05
) -> List[List[float]]:
    if mins.shape != (3,) or maxs.shape != (3,):
        raise ValueError("mins and maxs must be 3D vectors.")
    center = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins)
    half = np.maximum(half, 1e-3)
    half *= 1.0 + padding
    return [[center[i] - half[i], center[i] + half[i]] for i in range(3)]


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
    verts: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    ranges: List[List[float]],
    progress: float,
    camera: Dict[str, Dict[str, float]],
    center: np.ndarray,
    title_prefix: str,
    vertex_colors: Sequence[str],
    render_style: str,
) -> go.Figure:
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]])
    shifted = verts - center
    if render_style == "surface":
        fig.add_trace(
            go.Mesh3d(
                x=shifted[:, 0],
                y=shifted[:, 1],
                z=shifted[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=list(vertex_colors),
                flatshading=False,
                lighting=dict(
                    ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0
                ),
                name="Template→Target",
                showscale=False,
                hoverinfo="skip",
            )
        )
    else:
        coords = np.empty((edges.shape[0] * 3, 3), dtype=np.float32)
        coords[0::3] = shifted[edges[:, 0]]
        coords[1::3] = shifted[edges[:, 1]]
        coords[2::3] = np.nan
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="lines",
                line=dict(color="rgba(60, 60, 60, 0.85)", width=2),
                hoverinfo="skip",
                name="Wireframe",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=shifted[:, 0],
                y=shifted[:, 1],
                z=shifted[:, 2],
                mode="markers",
                marker=dict(color=list(vertex_colors), size=4, opacity=0.95),
                hoverinfo="skip",
                name="Vertices",
            )
        )
    fig.update_layout(
        title=f"{title_prefix} • progress → {progress*100:.1f}%",
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
    fig.add_annotation(
        text="Vertex colors track template indices",
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.02,
        showarrow=False,
        font=dict(size=12),
    )
    return fig


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
    latent_path = resolve_latent_path(
        args.latent_path,
        args.latent_root,
        args.latent_status,
        args.latent_index,
    ).resolve()
    print(f"Using latent code: {latent_path}")

    run_stem = f"{latent_path.stem}_morph"
    output_dir = ensure_output_dir(output_root, run_stem)
    decoder, template, _ = load_checkpoint(args.checkpoint, device)
    latent = load_latent_tensor(latent_path)

    if args.steps < 2:
        raise ValueError("Steps must be at least 2 to form a transition.")

    target_mesh = decode_latent(decoder, template, latent, device)
    template_verts = template.verts_packed().detach().cpu().numpy()
    target_verts = target_mesh.verts_packed().cpu().numpy()
    faces = template.faces_packed().detach().cpu().numpy().astype(np.int32)
    edges = template.edges_packed().detach().cpu().numpy().astype(np.int32)
    vertex_colors: Tuple[str, ...] = tuple(spherical_vertex_colors(template_verts))

    template_min, template_max = mesh_bounds(template)
    target_min, target_max = mesh_bounds(target_mesh)
    bounds_min = np.minimum(template_min, target_min)
    bounds_max = np.maximum(template_max, target_max)
    center = 0.5 * (bounds_min + bounds_max)
    template_rel = template_verts - center
    target_rel = target_verts - center
    ranges = axis_ranges(bounds_min - center, bounds_max - center, padding=0.05)
    max_radius = max(
        float(np.linalg.norm(template_rel, axis=1).max()),
        float(np.linalg.norm(target_rel, axis=1).max()),
    )
    radius = max(max_radius * 1.5, 0.5)

    title_prefix = f"{latent_path.stem} • template→target"

    frame_paths: List[Path] = []
    interpolated = interpolate_vertices(template_verts, target_verts, args.steps)
    for idx, (progress, verts) in enumerate(
        tqdm(interpolated, desc="Rendering frames", total=len(interpolated))
    ):
        if args.rotate_camera:
            angle = 2.0 * np.pi * idx / max(len(interpolated) - 1, 1)
        else:
            angle = 0.0
        camera = orbit_camera(angle, radius)
        fig = figure_for_mesh(
            verts,
            faces,
            edges,
            ranges,
            progress,
            camera,
            center,
            title_prefix,
            vertex_colors,
            args.render_style,
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


