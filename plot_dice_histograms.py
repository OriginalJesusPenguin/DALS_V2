#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    csv_path = Path(
        # "/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/results/liver_segments_vs_gt_overlap.csv"
        # "/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/results/ts_vs_gt_overlap.csv"
        "/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/results/ts_liver_vs_gt_overlap.csv"
    )
    df = pd.read_csv(csv_path)

    bins = np.linspace(0.0, 1.0, 26)

    plt.figure(figsize=(8, 4))
    plotting_order = [
        ("cirrhotic", "red", 0.5, 1),
        ("healthy", "blue", 0.6, 2),
    ]
    for status, color, alpha, zorder in plotting_order:
        subset = df[df["health_status"] == status]["dice"].values
        plt.hist(
            subset,
            bins=bins,
            alpha=alpha,
            label=status.capitalize(),
            color=color,
            edgecolor="black",
            zorder=zorder,
        )

    plt.xlabel("Dice coefficient")
    plt.ylabel("Count")
    plt.title("Liver segment vs GT Dice distribution")
    plt.legend()
    plt.tight_layout()

    out_path = csv_path.with_name("liver_overlap_hist_TS_GT.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved histogram to {out_path}")


if __name__ == "__main__":
    main()

