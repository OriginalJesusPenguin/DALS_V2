#!/usr/bin/env python3
"""Generate a pairplot of selected RELAX metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


METRIC_MAP = {
    "F1@0.01": "F1@0.01",
    "F1@0.02": "F1@0.02",
    "ChamferL2": "ChamferL2_x_10000",
    "EMD": "emd",
    "Laplace-Beltrami Eigenvalues L2": "lb_l2_delta",
    "Hausdorff": "Hausdorff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/home/ralbe/DALS/mesh_autodecoder/inference_scripts/relax_metrics.csv"),
        help="Path to the RELAX metrics CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ralbe/DALS/mesh_autodecoder/relax_explanations/relax_metrics_pairplot.png"),
        help="Destination for the generated pairplot PNG.",
    )
    parser.add_argument(
        "--hist-output",
        type=Path,
        default=Path("/home/ralbe/DALS/mesh_autodecoder/relax_explanations/relax_metrics_chamfer_hist.png"),
        help="Destination for ChamferL2 histogram panel PNG.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=2.2,
        help="Height of each subplot in the pairplot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    missing = [col for col in METRIC_MAP.values() if col not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns in CSV: {missing}")

    pairplot_df = df[list(METRIC_MAP.values())].rename(columns={v: k for k, v in METRIC_MAP.items()})

    sns.set(style="whitegrid")
    grid = sns.pairplot(
        pairplot_df,
        corner=False,
        height=args.height,
        diag_kind="hist",
        plot_kws={"s": 4.0, "alpha": 0.5},
    )
    grid.fig.suptitle("RELAX Metrics Pairplot", y=1.02)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)
    print(f"Saved pairplot to {args.output}")

    chamfer_col = METRIC_MAP["ChamferL2"]
    chamfer_vals = df[chamfer_col].to_numpy(dtype=float)
    finite = chamfer_vals[np.isfinite(chamfer_vals)]
    if finite.size == 0:
        raise ValueError("No finite ChamferL2 values available for histogram plotting.")

    chamfer_min = finite.min()
    chamfer_max = finite.max()
    if chamfer_max == chamfer_min:
        raise ValueError("ChamferL2 values are constant; cannot rescale to similarity.")

    similarity = (chamfer_max - chamfer_vals) / (chamfer_max - chamfer_min)
    similarity = similarity * 2.0 - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=False)
    sns.histplot(finite, bins=40, ax=axes[0], color="#2a9d8f")
    axes[0].set_title("ChamferL2 Histogram")
    axes[0].set_xlabel("ChamferL2")
    axes[0].set_ylabel("Count")

    sns.histplot(similarity, bins=40, ax=axes[1], color="#e76f51")
    axes[1].set_title("ChamferL2 Rescaled Similarity Histogram")
    axes[1].set_xlabel("Similarity (scaled to [-1, 1])")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    args.hist_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.hist_output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Chamfer histograms to {args.hist_output}")


if __name__ == "__main__":
    main()


