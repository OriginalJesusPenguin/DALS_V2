import os
import sys
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mpl_colors


def load_data(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def map_categories(series, order):
    order = list(order)
    mapping = {cat: idx for idx, cat in enumerate(order)}
    values = series.map(mapping)
    return values, mapping


def normalize_numeric(series):
    vals = series.astype(float).values
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax):
        # Avoid divide-by-zero; return zeros
        return np.zeros_like(vals), (vmin, vmax)
    normed = (vals - vmin) / (vmax - vmin)
    return normed, (vmin, vmax)


def draw_parallel_plot(df, title, out_path):
    # Configure seaborn style
    sns.set_theme(style="whitegrid")

    # Prepare axes (categorical + numeric)
    # 1) augmentation_type: map 'noaug'->'no', 'aug'->'yes' for display, and positions [0,1]
    aug_display = df['augmentation_type'].replace({'noaug': 'no', 'aug': 'yes'})
    aug_vals, aug_map = map_categories(aug_display, order=['no', 'yes'])

    # 2) scaling_type: individual, global -> positions
    scaling_vals, scaling_map = map_categories(df['scaling_type'], order=['individual', 'global'])

    # 3) decoder_mode: gcnn, mlp -> positions (keep encountered order but prefer gcnn, mlp)
    dec_order = ['gcnn', 'mlp']
    dec_vals, dec_map = map_categories(df['decoder_mode'], order=dec_order)

    # 4) latent_dim: treat as categorical, sorted unique
    latent_unique = sorted(df['latent_dim'].dropna().unique().tolist())
    latent_vals, latent_map = map_categories(df['latent_dim'], order=latent_unique)

    # 5) ChamferL2 x 10000_mean: numeric, normalize
    chamfer_col = 'ChamferL2 x 10000_mean'
    chamfer_vals, (ch_min, ch_max) = normalize_numeric(df[chamfer_col])
    chamfer_raw = df[chamfer_col].astype(float).values

    # X positions for each axis
    x_positions = np.arange(5)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw vertical axes
    for x in x_positions:
        ax.axvline(x=x, color='lightgray', linewidth=1.0, zorder=0)

    # Prepare y values per row across axes
    y_matrix = np.vstack([aug_vals, scaling_vals, dec_vals, latent_vals, chamfer_vals]).astype(float).T

    # Plot each row as a line
    # Give each line a distinct color based on raw Chamfer value
    cmap = cm.get_cmap('viridis')
    norm = mpl_colors.Normalize(vmin=np.nanmin(chamfer_raw), vmax=np.nanmax(chamfer_raw))
    for row_idx, yvals in enumerate(y_matrix):
        color = cmap(norm(chamfer_raw[row_idx]))
        ax.plot(x_positions, yvals, color=color, alpha=0.6, linewidth=1.1)

    # Tick labels for categorical axes
    def set_cat_ticks(x, mapping, display_labels):
        inv = {v: k for k, v in mapping.items()}
        ticks = sorted(inv.keys())
        labels = [display_labels.get(inv[t], str(inv[t])) if isinstance(display_labels, dict) else inv[t] for t in ticks]
        ax.set_yticks([])  # Hide global y ticks; we add per-axis labels using text
        ax.set_xticks(x_positions)

    # Add per-axis y tick labels as text
    def add_axis_labels(x, labels):
        # labels: list of strings in order of vertical positions [0..n-1]
        n = len(labels)
        for i, lab in enumerate(labels):
            ax.text(x - 0.02, i, lab, va='center', ha='right', fontsize=10, color='dimgray')

    # Axis 0: augmentation (no/yes)
    ax.text(0, 1.05, 'augmentation', ha='center', va='bottom', fontsize=12, fontweight='bold')
    add_axis_labels(0, ['no', 'yes'])

    # Axis 1: scaling_type
    ax.text(1, 1.05, 'scaling', ha='center', va='bottom', fontsize=12, fontweight='bold')
    add_axis_labels(1, ['individual', 'global'])

    # Axis 2: decoder_mode
    ax.text(2, 1.05, 'decoder', ha='center', va='bottom', fontsize=12, fontweight='bold')
    add_axis_labels(2, dec_order)

    # Axis 3: latent_dim
    ax.text(3, 1.05, 'latent_dim', ha='center', va='bottom', fontsize=12, fontweight='bold')
    add_axis_labels(3, [str(v) for v in latent_unique])

    # Axis 4: Chamfer (normalized) with legend for scale
    ax.text(4, 1.05, f"ChamferL2 x 10000 (norm)\n[{ch_min:.1f} .. {ch_max:.1f}]", ha='center', va='bottom', fontsize=12, fontweight='bold')
    # Add a few tick reference labels for 0, 0.5, 1 scaled
    for yv, lab in zip([0.0, 0.5, 1.0], [f"{ch_min:.0f}", f"{(ch_min + 0.5*(ch_max-ch_min)):.0f}", f"{ch_max:.0f}"]):
        ax.text(4 + 0.02, yv, lab, va='center', ha='left', fontsize=9, color='dimgray')

    # Cosmetics
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(-0.2, max(1.0, np.nanmax(y_matrix[:, :4]) + 0.2))
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['augmentation', 'scaling', 'decoder', 'latent_dim', 'Chamfer'], fontsize=11)
    sns.despine(left=True, bottom=False)

    # Add colorbar to explain per-line colors (by Chamfer raw values)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('ChamferL2 x 10000 (raw)', rotation=90)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    # Default path relative to repository root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(repo_root, 'model_inference_summary.csv')

    df = load_data(csv_path)

    # Ensure required columns exist
    required_cols = [
        'split_type', 'augmentation_type', 'scaling_type', 'decoder_mode', 'latent_dim',
        'ChamferL2 x 10000_mean'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Drop rows with missing critical values
    df = df.dropna(subset=['split_type', 'augmentation_type', 'scaling_type', 'decoder_mode', 'latent_dim', 'ChamferL2 x 10000_mean'])

    # Create output directory for figures
    out_dir = os.path.join(repo_root, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    # Plot for split_type = 'separate'
    df_sep = df[df['split_type'] == 'separate'].copy()
    if len(df_sep) > 0:
        out_path_sep = os.path.join(out_dir, 'parallel_coords_separate.png')
        draw_parallel_plot(df_sep, title="Parallel Coordinates (split_type = 'separate')", out_path=out_path_sep)
        print(f"Saved: {out_path_sep}")
    else:
        print("No rows for split_type = 'separate'")

    # Plot for split_type = 'mixed'
    df_mix = df[df['split_type'] == 'mixed'].copy()
    if len(df_mix) > 0:
        out_path_mix = os.path.join(out_dir, 'parallel_coords_mixed.png')
        draw_parallel_plot(df_mix, title="Parallel Coordinates (split_type = 'mixed')", out_path=out_path_mix)
        print(f"Saved: {out_path_mix}")
    else:
        print("No rows for split_type = 'mixed'")


if __name__ == '__main__':
    main()


