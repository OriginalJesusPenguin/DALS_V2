import numpy as np
import pandas as pd 
import seaborn as sns
import torch
import warnings
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pathlib import Path

# --- Paths ---
healthy_idxs = [1, 12, 18, 19, 27, 29, 30, 63, 72]
cirrhotic_idxs = [26, 58, 115, 158, 168, 175, 195, 284, 303, 416]

RESULTS_DIR = Path("/home/ralbe/DALS/mesh_autodecoder/inference_results/meshes_MeshDecoderTrainer_2025-11-06_12-00-26")
LATENTS_DIR = RESULTS_DIR / "latents"
EXPLANATIONS_DIR = Path("/home/ralbe/DALS/mesh_autodecoder/relax_explanations")
metrics_path = EXPLANATIONS_DIR / "relax_metrics_old.csv"
masks_path = EXPLANATIONS_DIR / "relax_masks_old.pt"
per_sample_metrics_path = RESULTS_DIR / "test_metrics_per_sample.csv"


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sns.set(style="whitegrid")

# --- Load reusable data ---
df = pd.read_csv(metrics_path)
masks_tensor = torch.load(masks_path, map_location="cpu")
per_sample_metrics_df = pd.read_csv(per_sample_metrics_path)

if not isinstance(masks_tensor, torch.Tensor):
    raise TypeError("Expected masks to be a torch.Tensor")

num_masks, latent_dim = masks_tensor.shape
print(f"Loaded masks: {num_masks} x {latent_dim}")

blue_red = sns.color_palette("coolwarm", as_cmap=True)
cmap_latent = blue_red
cmap_importance = blue_red
cmap_uncertainty = blue_red
mesh_cmap = LinearSegmentedColormap.from_list("blue_to_pink", ["#264b96", "#f26bbb"])


def render_row(ax, values, title, cmap, y_limits=None):
    norm = plt.Normalize(vmin=np.min(values), vmax=np.max(values))
    colors = cmap(norm(values))
    ax.bar(
        np.arange(latent_dim),
        values,
        color=colors,
        edgecolor="black",
        width=0.8,
    )
    ax.set_xlim(-0.5, latent_dim - 0.5)
    ax.set_xticks(np.arange(0, latent_dim, 8))
    ax.set_xticklabels(np.arange(0, latent_dim, 8), rotation=45)
    ax.set_ylabel("")
    ax.set_title(title)
    ax.grid(False)
    if y_limits is not None:
        ymin, ymax = y_limits
        if ymin is not None and ymax is not None and ymin != ymax:
            ax.set_ylim(ymin, ymax)
        else:
            span = np.max(values) - np.min(values)
            lower = np.min(values) - 0.05 * span
            upper = np.max(values) + 0.05 * span
            ax.set_ylim(lower, upper)
    return norm


def load_mesh_vertices_faces(path: Path):
    verts, faces, _ = load_obj(path.as_posix(), load_textures=False)
    faces_idx = faces.verts_idx.numpy() if isinstance(faces.verts_idx, torch.Tensor) else faces.verts_idx
    return verts.numpy(), faces_idx


def face_vertical_components(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = verts[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_norm[normals_norm == 0] = 1.0
    normals_dir = normals / normals_norm
    mean_normal = normals_dir.mean(axis=0)
    if np.dot(mean_normal, np.array([0.0, 0.0, 1.0])) < 0:
        normals_dir = -normals_dir
    return normals_dir[:, 2]


def draw_wireframe(ax, vertices, faces, title, vertical_range=None):
    mesh = Meshes([torch.tensor(vertices)], [torch.tensor(faces)])
    verts_np = mesh.verts_list()[0].numpy()
    faces_np = mesh.faces_list()[0].numpy()

    tris = verts_np[faces_np]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_norm[normals_norm == 0] = 1.0
    normals_dir = normals / normals_norm
    mean_normal = normals_dir.mean(axis=0)
    if np.dot(mean_normal, np.array([0.0, 0.0, 1.0])) < 0:
        normals_dir = -normals_dir
    vertical_component = normals_dir[:, 2]
    if vertical_range is None:
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = vertical_range
    normed = (vertical_component - vmin) / (vmax - vmin + 1e-8)
    color_vals = mesh_cmap(np.clip(normed, 0.0, 1.0))

    poly = Poly3DCollection(
        tris,
        facecolor=color_vals,
        edgecolor=(0, 0, 0, 0.15),
        linewidths=0.03,
        alpha=0.9,
    )
    ax.add_collection3d(poly)
    ax.set_title(title)
    ax.set_axis_off()
    mins = verts_np.min(axis=0)
    maxs = verts_np.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = (maxs - mins).max() / 2.0
    ax.set_xlim(center[0] - scale, center[0] + scale)
    ax.set_ylim(center[1] - scale, center[1] + scale)
    ax.set_zlim(center[2] - scale, center[2] + scale)


def fetch_metric(row, keys, default="n/a", fmt="{:.4f}"):
    for key in keys:
        if key in row:
            value = row[key]
            try:
                if isinstance(value, str):
                    return value
                if np.isnan(value):
                    return default
                return fmt.format(value)
            except Exception:
                return value
    return default


def process_sample(status: str, idx: int):
    target_latent = f"{status}_{idx}_testing"
    latent_path = LATENTS_DIR / f"{target_latent}_latent.pt"
    target_mesh_path = RESULTS_DIR / f"{target_latent}_target.obj"
    pred_mesh_path = RESULTS_DIR / f"{target_latent}_optimized.obj"

    if not latent_path.exists():
        print(f"Skipping {target_latent}: latent file not found.")
        return
    if not target_mesh_path.exists() or not pred_mesh_path.exists():
        print(f"Skipping {target_latent}: mesh file(s) missing.")
        return

    latent_tensor = torch.load(latent_path.as_posix(), map_location="cpu")
    if latent_tensor.ndim != 1:
        latent_tensor = latent_tensor.view(-1)
    latents_np = latent_tensor.numpy()

    sample_df = df[df["latent_name"] == target_latent].copy()
    if sample_df.empty:
        print(f"Skipping {target_latent}: no rows in metrics CSV.")
        return
    print(f"\nProcessing {target_latent}: {len(sample_df)} rows")

    if "mask_index" in sample_df.columns:
        sample_df = sample_df.sort_values("mask_index").reset_index(drop=True)
        mask_indices = sample_df["mask_index"].to_numpy(dtype=int)
    else:
        sample_df = sample_df.reset_index(drop=True)
        mask_indices = np.arange(len(sample_df), dtype=int)
        print("  mask_index column not found; assuming sorted order.")

    if mask_indices.max() >= num_masks:
        print(f"  Skipping {target_latent}: mask indices exceed available mask count.")
        return

    sample_masks = masks_tensor[mask_indices].numpy()
    if sample_masks.shape[0] != len(sample_df):
        print(f"  Skipping {target_latent}: mismatch between masks and rows.")
        return

    if "ChamferL2_x_10000" not in sample_df.columns:
        print(f"  Skipping {target_latent}: Chamfer column missing.")
        return

    distances = sample_df["ChamferL2_x_10000"].to_numpy(dtype=float)
    finite_dist = distances[np.isfinite(distances)]
    if len(finite_dist) == 0:
        print(f"  Skipping {target_latent}: no finite Chamfer distances.")
        return

    min_d = np.min(finite_dist)
    max_d = np.max(finite_dist)
    if not np.isfinite(min_d) or not np.isfinite(max_d) or max_d == min_d:
        print(f"  Skipping {target_latent}: degenerate Chamfer range.")
        return

    similarity = (max_d - distances) / (max_d - min_d + 1e-12) * 2.0 - 1.0
    similarity = np.clip(similarity, -1.0, 1.0)
    sample_df["similarity"] = similarity

    similarity_column = similarity.reshape(-1, 1)
    importance = (similarity_column * sample_masks).mean(axis=0)
    diff = similarity_column - importance.reshape(1, -1)
    uncertainty = (np.square(diff) * sample_masks).mean(axis=0)

    print("  Importance shape:", importance.shape)

    importance_df = pd.DataFrame(
        {
            "latent_index": np.arange(latent_dim, dtype=int),
            "importance": importance,
            "uncertainty": uncertainty,
        }
    )

    target_verts, target_faces = load_mesh_vertices_faces(target_mesh_path)
    pred_verts, pred_faces = load_mesh_vertices_faces(pred_mesh_path)

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(6, 2, height_ratios=[2, 2, 2, 0.3, 12, 1.3], width_ratios=[40, 1], hspace=0.6, wspace=0.3)

    ax_latent = fig.add_subplot(gs[0, 0])
    lat_span = np.max(latents_np) - np.min(latents_np)
    lat_limits = (np.min(latents_np) - 0.05 * lat_span, np.max(latents_np) + 0.05 * lat_span)
    norm_latent = render_row(ax_latent, latents_np, "Latent Values", cmap_latent, y_limits=lat_limits)
    cbar_latent = fig.add_subplot(gs[0, 1])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_latent, cmap=cmap_latent), cax=cbar_latent)

    ax_imp = fig.add_subplot(gs[1, 0])
    imp_span = np.max(importance) - np.min(importance)
    imp_limits = (np.min(importance) - 0.05 * imp_span, np.max(importance) + 0.05 * imp_span)
    norm_imp = render_row(ax_imp, importance, r"Importance $R_i$", cmap_importance, y_limits=imp_limits)
    cbar_imp = fig.add_subplot(gs[1, 1])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_imp, cmap=cmap_importance), cax=cbar_imp)

    ax_unc = fig.add_subplot(gs[2, 0])
    unc_span = np.max(uncertainty) - np.min(uncertainty)
    unc_limits = (np.min(uncertainty) - 0.05 * unc_span, np.max(uncertainty) + 0.05 * unc_span)
    norm_unc = render_row(ax_unc, uncertainty, r"Uncertainty $U_i$", cmap_uncertainty, y_limits=unc_limits)
    cbar_unc = fig.add_subplot(gs[2, 1])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_unc, cmap=cmap_uncertainty), cax=cbar_unc)

    mesh_grid = gs[4, 0].subgridspec(2, 3, wspace=0.05, hspace=0.05)
    target_verticals = face_vertical_components(target_verts, target_faces)
    pred_verticals = face_vertical_components(pred_verts, pred_faces)
    vmin = min(np.min(target_verticals), np.min(pred_verticals))
    vmax = max(np.max(target_verticals), np.max(pred_verticals))

    view_specs = [
        ("Anterior", 20, 45),
        ("Posterior", 20, 225),
        ("Superior", 80, -90),
    ]

    for col, (label, elev, azim) in enumerate(view_specs):
        ax_target = fig.add_subplot(mesh_grid[0, col], projection="3d")
        draw_wireframe(ax_target, target_verts, target_faces, f"Target – {label}", vertical_range=(vmin, vmax))
        ax_target.view_init(elev=elev, azim=azim)

    for col, (label, elev, azim) in enumerate(view_specs):
        ax_pred = fig.add_subplot(mesh_grid[1, col], projection="3d")
        draw_wireframe(ax_pred, pred_verts, pred_faces, f"Decoded – {label}", vertical_range=(vmin, vmax))
        ax_pred.view_init(elev=elev, azim=azim)

    target_mesh_basenames = [
        target_mesh_path.name,
        target_mesh_path.name.replace("_target", ""),
        f"{target_latent}.obj",
        target_latent,
    ]
    metrics_row = None
    for candidate_col in [
        "target_mesh_name",
        "target_name",
        "mesh_name",
        "optimized_mesh_name",
        "latent_name",
        "sample",
        "sample_name",
        "id",
    ]:
        if candidate_col in per_sample_metrics_df.columns:
            subset = per_sample_metrics_df[per_sample_metrics_df[candidate_col].isin(target_mesh_basenames)]
            if not subset.empty:
                metrics_row = subset.iloc[0]
                break
    if metrics_row is None:
        metrics_row = per_sample_metrics_df.iloc[0]
        print("  Warning: target sample not found in per-sample metrics; using first row.")

    raw_no_ints = fetch_metric(metrics_row, ['No. ints.', 'No_ints_percent', 'No_ints'], fmt="{:.2f}")
    if isinstance(raw_no_ints, str) and raw_no_ints != "n/a" and not raw_no_ints.endswith("%"):
        no_ints_display = f"{raw_no_ints}%"
    else:
        no_ints_display = raw_no_ints

    metrics_lines = [
        f"ChamferL2: {fetch_metric(metrics_row, ['ChamferL2_x_10000', 'ChamferL2 x 10000', 'ChamferL2'])}",
        f"Quality: {fetch_metric(metrics_row, ['BL quality', 'quality', 'Quality'])}",
        f"No. Ints: {no_ints_display}",
        f"F1@0.01: {fetch_metric(metrics_row, ['F1@0.01', 'F1_0.01'])}",
        f"F1@0.02: {fetch_metric(metrics_row, ['F1@0.02', 'F1_0.02'])}",
        f"Hausdorff: {fetch_metric(metrics_row, ['Hausdorff'])}",
    ]

    ax_metrics = fig.add_subplot(gs[5, :])
    ax_metrics.axis("off")
    ax_metrics.text(
        0.5,
        0.6,
        "   |   ".join(str(item) for item in metrics_lines),
        ha="center",
        va="center",
        fontsize=13,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()

    csv_path = EXPLANATIONS_DIR / f"importance_{target_latent}.csv"
    panel_path = EXPLANATIONS_DIR / f"importance_uncertainty_panel_{target_latent}.png"
    importance_df.to_csv(csv_path, index=False)
    fig.savefig(panel_path, dpi=300)
    plt.close(fig)
    print(f"  Saved CSV to {csv_path}")
    print(f"  Saved panel to {panel_path}")


for status, idxs in [("healthy", healthy_idxs), ("cirrhotic", cirrhotic_idxs)]:
    for sample_idx in idxs:
        process_sample(status, sample_idx)