#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from tqdm import tqdm

BASE_DIR = Path(
    "/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/processed"
)
RESULTS_DIR = BASE_DIR.parent / "results"

DATASETS = ("healthy", "cirrhotic")


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.count_nonzero(mask_a & mask_b)
    size_a = np.count_nonzero(mask_a)
    size_b = np.count_nonzero(mask_b)
    if size_a + size_b == 0:
        return 1.0
    return 2.0 * intersection / (size_a + size_b)


def load_binary_mask(path: Path) -> nib.Nifti1Image:
    img = nib.load(str(path))
    data = img.get_fdata()
    binary = (data > 0.5).astype(np.uint8)
    return nib.Nifti1Image(binary, img.affine, img.header)


def ensure_alignment(source: nib.Nifti1Image, target: nib.Nifti1Image) -> nib.Nifti1Image:
    if source.shape == target.shape and np.allclose(source.affine, target.affine):
        return target
    return resample_from_to(target, source, order=0)


def gather_segmentations() -> Dict[str, Dict[str, Path]]:
    segmentations: Dict[str, Dict[str, Path]] = {}
    for health_status in DATASETS:
        seg_root = BASE_DIR / health_status / "T1_images"
        patient_to_seg: Dict[str, Path] = {}
        for seg_dir in sorted(seg_root.glob("segmentations_*")):
            if not seg_dir.is_dir():
                continue
            patient_id = seg_dir.name.split("_", 1)[-1]
            seg_path = seg_dir / "liver_segments_merged.nii.gz"
            if seg_path.is_file():
                patient_to_seg[patient_id] = seg_path
        segmentations[health_status] = patient_to_seg
    return segmentations


def iter_gt_vs_ts() -> List[Tuple[str, str, Path, Path]]:
    pairs: List[Tuple[str, str, Path, Path]] = []
    for health_status in DATASETS:
        gt_root = BASE_DIR / health_status / "T1_masks" / "GT"
        ts_root = BASE_DIR / health_status / "T1_masks" / "TS"
        for gt_path in sorted(gt_root.glob("*.nii.gz")):
            patient_id = gt_path.stem
            ts_path = ts_root / patient_id / "liver.nii.gz"
            if not ts_path.is_file():
                continue
            pairs.append((patient_id, health_status, gt_path, ts_path))
    return pairs


def iter_gt_vs_segments(segmentations: Dict[str, Dict[str, Path]]) -> List[Tuple[str, str, Path, Path]]:
    pairs: List[Tuple[str, str, Path, Path]] = []
    for health_status in DATASETS:
        mapping = segmentations.get(health_status, {})
        gt_root = BASE_DIR / health_status / "T1_masks" / "GT"
        for patient_id, seg_path in mapping.items():
            gt_path = gt_root / f"{patient_id}.nii.gz"
            if not gt_path.is_file():
                continue
            pairs.append((patient_id, health_status, gt_path, seg_path))
    return pairs


def iter_ts_vs_segments(segmentations: Dict[str, Dict[str, Path]]) -> List[Tuple[str, str, Path, Path]]:
    pairs: List[Tuple[str, str, Path, Path]] = []
    for health_status in DATASETS:
        mapping = segmentations.get(health_status, {})
        ts_root = BASE_DIR / health_status / "T1_masks" / "TS"
        for patient_id, seg_path in mapping.items():
            ts_path = ts_root / patient_id / "liver.nii.gz"
            if not ts_path.is_file():
                continue
            pairs.append((patient_id, health_status, ts_path, seg_path))
    return pairs


def compute_overlaps(
    name: str,
    pairs: Iterable[Tuple[str, str, Path, Path]],
    output_path: Path,
    force: bool,
) -> pd.DataFrame:
    if output_path.is_file() and not force:
        print(f"[SKIP] {name}: found existing {output_path}")
        return pd.read_csv(output_path)

    rows = []
    pairs_list = list(pairs)
    progress = tqdm(pairs_list, desc=f"{name}", unit="pair")
    for patient_id, health_status, ref_path, target_path in progress:
        ref_img = load_binary_mask(ref_path)
        target_img = load_binary_mask(target_path)
        aligned_target = ensure_alignment(ref_img, target_img)

        ref_data = ref_img.get_fdata().astype(bool)
        target_data = aligned_target.get_fdata().astype(bool)
        dice = dice_coefficient(ref_data, target_data)

        rows.append(
            {
                "patient_id": patient_id,
                "health_status": health_status,
                "dice": dice,
            }
        )
        progress.set_postfix(dice=f"{dice:.3f}")

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[WRITE] {name}: saved {output_path} ({len(df)} pairs)")
    return df


def plot_histograms(
    datasets: List[Tuple[str, pd.DataFrame]],
    output_path: Path,
) -> None:
    bins = np.linspace(0.0, 1.0, 26)
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8, 10), sharex=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, (title, df) in zip(axes, datasets):
        for status, color, alpha in (
            ("cirrhotic", "#EF553B", 0.6),
            ("healthy", "#636EFA", 0.6),
        ):
            subset = df[df["health_status"] == status]["dice"].values
            if subset.size == 0:
                continue
            ax.hist(
                subset,
                bins=bins,
                alpha=alpha,
                label=status.capitalize(),
                color=color,
                edgecolor="black",
            )
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()

    axes[-1].set_xlabel("Dice coefficient")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[WRITE] Saved histogram figure to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Dice overlaps for GT, TS, and liver segments and plot histograms."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute CSV files even if they already exist.",
    )
    parser.add_argument(
        "--histogram-path",
        type=Path,
        default=RESULTS_DIR / "liver_overlap_histograms.png",
        help="Path for the combined histogram figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    segmentations = gather_segmentations()

    gt_ts_pairs = iter_gt_vs_ts()
    gt_seg_pairs = iter_gt_vs_segments(segmentations)
    ts_seg_pairs = iter_ts_vs_segments(segmentations)

    gt_ts_csv = RESULTS_DIR / "ts_liver_vs_gt_overlap.csv"
    seg_gt_csv = RESULTS_DIR / "liver_segments_vs_gt_overlap.csv"
    seg_ts_csv = RESULTS_DIR / "liver_segments_vs_ts_overlap.csv"

    gt_ts_df = compute_overlaps("GT vs TS liver", gt_ts_pairs, gt_ts_csv, args.force)
    seg_gt_df = compute_overlaps("GT vs liver segments", gt_seg_pairs, seg_gt_csv, args.force)
    seg_ts_df = compute_overlaps("TS vs liver segments", ts_seg_pairs, seg_ts_csv, args.force)

    datasets = [
        ("GT vs TS liver Dice", gt_ts_df),
        ("GT vs liver segments Dice", seg_gt_df),
        ("TS vs liver segments Dice", seg_ts_df),
    ]
    plot_histograms(datasets, args.histogram_path)


if __name__ == "__main__":
    main()



