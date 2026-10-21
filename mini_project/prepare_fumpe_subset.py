"""Convert real FUMPE CTA scans and radiologist masks to a compact slice dataset.

Input FUMPE images/masks are NIfTI volumes. Output is 224px PNG images and a
manifest with labels produced directly from non-empty expert segmentation masks.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import nibabel as nib

from pe_screening import resize_image


def normalise_ct(slice_image: np.ndarray) -> np.ndarray:
    """Use a fixed pulmonary-angiography window rather than normalising each slice.

    Slice-wise percentile scaling changes CT attenuation values from one slice to
    another and suppresses the subtle contrast differences relevant to PE. The
    fixed 100/700 (level/width) window is a conventional CTPA visualisation.
    """
    low, high = -250.0, 450.0  # centre 100 HU, width 700 HU
    return np.clip((slice_image.astype(np.float32) - low) / (high - low), 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced real PE/no-PE slices from FUMPE expert masks.")
    parser.add_argument("--fumpe-root", default="data/raw/fumpe")
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--positive-per-study", type=int, default=25, help="Cap positives per patient to keep training quick")
    parser.add_argument("--negative-per-positive", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.fumpe_root)
    images_dir, masks_dir = root / "images", root / "masks"
    image_paths = sorted(images_dir.glob("PAT*.nii.gz"))
    if not image_paths:
        raise SystemExit(f"No FUMPE image volumes found in {images_dir}. Run download_fumpe_subset.py first.")
    output = Path(args.output); output_images = output / "images"; output_images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    for image_path in image_paths:
        patient = image_path.name.replace(".nii.gz", "")
        mask_path = masks_dir / image_path.name
        if not mask_path.exists():
            print(f"Skipping {patient}: matching mask was not found")
            continue
        volume = np.asarray(nib.load(image_path).get_fdata(dtype=np.float32))
        mask = np.asarray(nib.load(mask_path).get_fdata(dtype=np.float32))
        if volume.shape != mask.shape or volume.ndim != 3:
            print(f"Skipping {patient}: incompatible image/mask shape {volume.shape} / {mask.shape}")
            continue
        positive = np.flatnonzero(np.any(mask > 0, axis=(0, 1)))
        negative = np.flatnonzero(~np.any(mask > 0, axis=(0, 1)))
        if len(positive) == 0:
            print(f"Skipping {patient}: no labelled PE slices")
            continue
        chosen_positive = rng.choice(positive, size=min(args.positive_per_study, len(positive)), replace=False)
        number_negative = min(len(negative), len(chosen_positive) * args.negative_per_positive)
        chosen_negative = rng.choice(negative, size=number_negative, replace=False)
        for label, indices in ((1, chosen_positive), (0, chosen_negative)):
            for index in indices:
                image = resize_image(normalise_ct(volume[:, :, int(index)]))
                filename = f"fumpe_{patient}_{int(index):04d}_label{label}.png"
                Image.fromarray((image * 255).astype(np.uint8)).save(output_images / filename, optimize=True)
                records.append({"image_path": f"images/{filename}", "label": label, "study_id": patient, "source": "FUMPE_expert_mask"})
        print(f"{patient}: {len(chosen_positive)} PE-positive slices, {number_negative} PE-negative slices")
    if not records:
        raise SystemExit("No usable records were created.")
    manifest = pd.DataFrame(records)
    manifest.to_csv(output / "manifest.csv", index=False)
    print(f"Created {len(manifest)} real labelled CT slices in {output}. Positives: {int(manifest.label.sum())}.")


if __name__ == "__main__":
    main()
