"""Create a tiny synthetic CT-like dataset so the project works before licensed data is added.

The images are simulated and are never medical data. Delete this dataset before evaluating
the project on real data.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def synthetic_slice(label: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = 224
    yy, xx = np.mgrid[:size, :size]
    radius = np.sqrt((xx - 112) ** 2 + (yy - 112) ** 2)
    image = np.clip(0.08 + rng.normal(0, 0.025, (size, size)), 0, 1)
    lungs = ((xx - 70) ** 2 / 45 ** 2 + (yy - 112) ** 2 / 74 ** 2 < 1) | ((xx - 154) ** 2 / 45 ** 2 + (yy - 112) ** 2 / 74 ** 2 < 1)
    image[lungs] = np.clip(0.28 + rng.normal(0, 0.035, lungs.sum()), 0, 1)
    image[radius > 105] *= 0.3
    # A synthetic vascular structure; class 1 contains a simulated filling defect.
    vessel = (xx - 112) ** 2 + (yy - 118) ** 2 < 19 ** 2
    image[vessel] = 0.78
    if label:
        cx, cy = rng.integers(103, 121), rng.integers(109, 127)
        defect = (xx - cx) ** 2 + (yy - cy) ** 2 < rng.integers(5, 9) ** 2
        image[defect] = 0.35
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--per-class", type=int, default=120)
    args = parser.parse_args()
    root = Path(args.output); image_dir = root / "images"; image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in (0, 1):
        for index in range(args.per_class):
            name = f"demo_{label}_{index:03d}.png"
            Image.fromarray(synthetic_slice(label, seed=label * 10000 + index)).save(image_dir / name)
            rows.append({"image_path": f"images/{name}", "label": label, "study_id": f"DEMO-{label}-{index:03d}", "source": "synthetic_demo_only"})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    print(f"Created {len(rows)} simulated images at {root}. Do not use them as medical evidence.")


if __name__ == "__main__":
    main()
