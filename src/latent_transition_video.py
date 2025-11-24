#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pytorch3d.structures import Meshes
# import pytorch3d.subdivide_meshes
from pytorch3d.ops import subdivide_meshes
from torch import nn

import imageio.v2 as imageio
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.mesh_decoder import MeshDecoder  # noqa: E402

DISTANCE_COLORMAP_MIN = 0.0
DISTANCE_COLORMAP_MAX = 0.25


def parse_args() -> argparse.Namespace:
    default_checkpoint = Path(
        "/home/ralbe/DALS/mesh_autodecoder/models/MeshDecoderTrainer_2025-11-06_12-00-26.ckpt"
    )
    default_latent_dir = Path(
        "/home/ralbe/DALS/mesh_autodecoder/inference_results/"
        "meshes_MeshDecoderTrainer_2025-11-06_12-00-26/latents"
    )
    default_latent_path = default_latent_dir / "cirrhotic_115_testing_latent.pt"
    parser = argparse.ArgumentParser(
        description="Render a latent transition as frame sequence and video."
    )
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument(
        "--latent-source",
        choices=["file", "cirrhotic", "healthy", "mean", "healthy_mean", "cirrhotic_mean"],
        default="file",
        help="Select how the base latent vector is obtained.",
    )
    parser.add_argument(
        "--latent-path",
        type=Path,
        default=default_latent_path,
        help="Latent tensor to load when --latent-source=file.",
    )
    parser.add_argument(
        "--latent-dir",
        type=Path,
        default=default_latent_dir,
        help="Directory containing latent tensors for cirrhotic/healthy selections.",
    )
    parser.add_argument(
        "--cohort-index",
        type=int,
        default=0,
        help="Index for selecting a latent within the chosen cohort (sorted ascending).",
    )
    parser.add_argument("--latent-index", type=int, required=False, default=66)
    parser.add_argument("--min-value", type=float, required=True)
    parser.add_argument("--max-value", type=float, required=True)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--render-mode",
        choices=["overlap", "distance_colormap"],
        default="overlap",
        help="Visualization mode for decoded meshes.",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "video_frames")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument(
        "--classifier-model",
        type=Path,
        default=Path(
            "/home/ralbe/DALS/mesh_autodecoder/inference_results/latent_classifier_outputs/best_model_seed_1343.pt"
        ),
        help="Path to a saved latent classifier checkpoint (.pt) for probability bars.",
    )
    return parser.parse_args()


def ensure_output_dir(base: Path, stem: str, latent_index: int) -> Path:
    run_stem = f"{stem}_idx{latent_index}"
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [
            path
            for path in base.iterdir()
            if path.is_dir() and path.name.startswith(f"{run_stem}_")
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
    out_dir = base / f"{run_stem}_{next_idx:04d}"
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


def select_cohort_latent(latent_dir: Path, cohort: str, index: int) -> Tuple[torch.Tensor, Path]:
    if not latent_dir.exists():
        raise FileNotFoundError(f"Latent directory not found: {latent_dir}")
    pattern = f"{cohort}_*_latent.pt"
    files = sorted(latent_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No '{cohort}' latents found in {latent_dir}")
    if index < 0 or index >= len(files):
        raise IndexError(
            f"Cohort index {index} out of range for '{cohort}' latents "
            f"(found {len(files)} files)."
        )
    path = files[index]
    latent = load_latent_tensor(path)
    return latent, path


def compute_mean_latent_from_dir(latent_dir: Path, cohort: Optional[str] = None) -> torch.Tensor:
    if not latent_dir.exists():
        raise FileNotFoundError(f"Latent directory not found: {latent_dir}")
    latents: List[torch.Tensor] = []
    pattern = f"{cohort}_*_latent.pt" if cohort is not None else "*_latent.pt"
    for path in sorted(latent_dir.glob(pattern)):
        try:
            latents.append(load_latent_tensor(path))
        except Exception as exc:
            print(f"[WARN] Skipping {path}: {exc}")
    if not latents:
        cohort_msg = f" for cohort '{cohort}'" if cohort is not None else ""
        raise RuntimeError(f"No latent tensors could be loaded from {latent_dir}{cohort_msg}")
    stacked = torch.stack(latents).float()
    return stacked.mean(dim=0)


def resolve_latent(
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, str, Optional[Path]]:
    source = args.latent_source
    if source == "file":
        if args.latent_path is None:
            raise ValueError("latent-path must be provided when --latent-source=file.")
        path = args.latent_path
        if not path.exists():
            raise FileNotFoundError(f"Latent file not found: {path}")
        latent = load_latent_tensor(path)
        descriptor = path.stem
        return latent, descriptor, path

    if source in ("cirrhotic", "healthy"):
        latent, path = select_cohort_latent(args.latent_dir, source, args.cohort_index)
        descriptor = path.stem
        return latent, descriptor, path

    if source == "mean":
        latent = compute_mean_latent_from_dir(args.latent_dir)
        descriptor = "latent_mean"
        return latent, descriptor, None

    if source in ("healthy_mean", "cirrhotic_mean"):
        cohort = source.split("_", 1)[0]
        latent = compute_mean_latent_from_dir(args.latent_dir, cohort=cohort)
        descriptor = f"{cohort}_mean_latent"
        return latent, descriptor, None

    raise ValueError(f"Unsupported latent source: {source}")


class LatentHealthClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def load_classifier(model_path: Path, device: torch.device) -> Optional[LatentHealthClassifier]:
    if model_path is None:
        return None
    if not model_path.exists():
        print(f"[WARN] Classifier checkpoint not found: {model_path}")
        return None
    payload = torch.load(model_path, map_location=device)
    state_dict = payload.get("state_dict")
    if state_dict is None:
        print(f"[WARN] No state_dict in classifier checkpoint: {model_path}")
        return None
    first_layer = state_dict.get("network.0.weight")
    if first_layer is None:
        print(f"[WARN] Unexpected classifier state format in {model_path}")
        return None
    hidden_dim, in_features = first_layer.shape
    classifier = LatentHealthClassifier(in_features=in_features, hidden_dim=hidden_dim)
    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()
    print(
        f"[INFO] Loaded latent classifier from {model_path} "
        f"(hidden_dim={hidden_dim}, input_dim={in_features})"
    )
    return classifier


def predict_cirrhosis_probability(
    classifier: LatentHealthClassifier, latent_vector: torch.Tensor, device: torch.device
) -> float:
    with torch.no_grad():
        latent_tensor = latent_vector.to(device=device, dtype=torch.float32).unsqueeze(0)
        logit = classifier(latent_tensor)
        prob = torch.sigmoid(logit).item()
    return float(prob)


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


def mesh_to_distance_trace(
    mesh: Meshes,
    distances: np.ndarray,
    center: np.ndarray,
    cmin: float,
    cmax: float,
) -> go.Mesh3d:
    verts = mesh.verts_packed().cpu().numpy() - center
    faces = mesh.faces_packed().cpu().numpy()
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=distances,
        cmin=cmin,
        cmax=cmax,
        colorscale="Turbo",
        colorbar=dict(title="Distance"),
        showscale=True,
        name="Modified",
        lighting=dict(ambient=0.7, diffuse=0.7),
        flatshading=True,
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
    original_value: float,
    class_probs: Optional[Tuple[float, float]] = None,
    render_mode: str = "overlap",
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        column_widths=[0.7, 0.3],
    )
    if render_mode == "overlap":
        fig.add_trace(
            mesh_to_wireframe_trace(original, "#636EFA", 1, "Original", center),
            row=1,
            col=1,
        )
        fig.add_trace(
            mesh_to_wireframe_trace(current, "#EF553B", 3, "Modified", center),
            row=1,
            col=1,
        )
    else:
        verts_orig = original.verts_packed().cpu().numpy()
        verts_cur = current.verts_packed().cpu().numpy()
        if verts_orig.shape != verts_cur.shape:
            raise ValueError(
                "Original and modified meshes must share the same vertex layout for distance visualization."
            )
        distances = np.linalg.norm(verts_cur - verts_orig, axis=1)
        distances = np.clip(distances, DISTANCE_COLORMAP_MIN, DISTANCE_COLORMAP_MAX)
        fig.add_trace(
            mesh_to_distance_trace(
                current,
                distances,
                center,
                DISTANCE_COLORMAP_MIN,
                DISTANCE_COLORMAP_MAX,
            ),
            row=1,
            col=1,
        )
    fig.update_layout(
        title=f"{title_prefix} from {original_value:+.4f} → {step_value:+.4f}",
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
    if class_probs is not None:
        healthy_prob, cirrhotic_prob = class_probs
        bar_trace = go.Bar(
            x=["Healthy (0)", "Cirrhotic (1)"],
            y=[healthy_prob * 100.0, cirrhotic_prob * 100.0],
            marker=dict(color=["#10B981", "#EF4444"]),
            text=[f"{healthy_prob * 100.0:.1f}%", f"{cirrhotic_prob * 100.0:.1f}%"],
            textposition="auto",
            hovertemplate="%{x}<br>%{y:.2f}%%<extra></extra>",
            name="Probability",
        )
        fig.add_trace(bar_trace, row=1, col=2)
        fig.update_yaxes(range=[0, 100], title="Probability (%)", row=1, col=2)
        fig.update_xaxes(tickangle=-20, row=1, col=2)

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
    latent, latent_descriptor, latent_path = resolve_latent(args)
    if latent_path is not None:
        print(f"[INFO] Using latent file: {latent_path}")
    else:
        print(f"[INFO] Using latent descriptor: {latent_descriptor}")
    output_dir = ensure_output_dir(output_root, latent_descriptor, args.latent_index)
    decoder, template, _ = load_checkpoint(args.checkpoint, device)

    classifier = load_classifier(args.classifier_model, device) if args.classifier_model else None

    if not 0 <= args.latent_index < latent.numel():
        raise IndexError(f"Latent index {args.latent_index} is out of bounds for tensor of size {latent.numel()}.")
    if args.steps < 2:
        raise ValueError("Steps must be at least 2 to form a transition.")

    original_value = float(latent[args.latent_index])
    original_mesh = decode_latent(decoder, template, latent, device)

    step_values = np.linspace(args.min_value, args.max_value, args.steps, dtype=np.float32)
    modified_meshes: List[Meshes] = []
    class_probabilities: List[Optional[Tuple[float, float]]] = []
    for value in step_values:
        modified = latent.clone()
        modified[args.latent_index] = float(value)
        mesh = decode_latent(decoder, template, modified, device)
        modified_meshes.append(mesh)
        if classifier is not None:
            prob_cirrhotic = predict_cirrhosis_probability(classifier, modified, device)
            class_probabilities.append((1.0 - prob_cirrhotic, prob_cirrhotic))
        else:
            class_probabilities.append(None)

    bounds_min, bounds_max = accumulate_bounds([original_mesh, *modified_meshes])
    center = 0.5 * (bounds_min + bounds_max)
    ranges = axis_ranges(bounds_min - center, bounds_max - center)
    span = max(r[1] - r[0] for r in ranges)
    radius = max(span * 1.6, 1.0)

    title_prefix = f"{latent_descriptor} • latent[{args.latent_index}]"

    frame_paths: List[Path] = []
    for idx, (value, mesh, probs) in enumerate(
        tqdm(
            list(zip(step_values, modified_meshes, class_probabilities)),
            desc="Rendering frames",
            unit="frame",
        )
    ):
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
            original_value,
            probs,
            args.render_mode,
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


