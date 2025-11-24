#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(
    "/home/ralbe/pyhppc_project/cirr_segm_clean/cirrhotic_liver_segmentation/cirr_mri_600/data/processed"
)


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


def main() -> None:
    rows = []
    for health_status in ["healthy", "cirrhotic"]:
        gt_root = BASE_DIR / health_status / "T1_masks" / "GT"
        ts_root = BASE_DIR / health_status / "T1_masks" / "TS"

        gt_files = sorted(gt_root.glob("*.nii.gz"))
        total_dice = 0.0
        count = 0
        progress = tqdm(gt_files, desc=f"{health_status} GT vs TS")

        for gt_path in progress:
            if not gt_path.name.endswith(".nii.gz"):
                continue
            patient_id = gt_path.name[:-7]
            ts_path = ts_root / patient_id / "liver.nii.gz"
            if not ts_path.is_file():
                continue

            gt_img = load_binary_mask(gt_path)
            ts_img = load_binary_mask(ts_path)

            aligned_ts_img = ensure_alignment(gt_img, ts_img)
            gt_data = gt_img.get_fdata().astype(bool)
            ts_data = aligned_ts_img.get_fdata().astype(bool)

            dice = dice_coefficient(gt_data, ts_data)
            total_dice += dice
            count += 1
            progress.set_postfix(last=f"{dice:.3f}", avg=f"{(total_dice / count):.3f}")

            rows.append(
                {
                    "patient_id": patient_id,
                    "health_status": health_status,
                    "dice": dice,
                }
            )

    df = pd.DataFrame(rows)
    output_path = BASE_DIR.parent / "results" / "ts_liver_vs_gt_overlap.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()

