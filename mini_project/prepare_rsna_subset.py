"""Make a storage-capped PNG subset from data the user is licensed to access.

Expected RSNA training label columns: SOPInstanceUID and pe_present_on_image.
"""
from pathlib import Path
import argparse
import pandas as pd
from PIL import Image
from pe_screening import load_image, resize_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced local RSNA subset. Does not download any data.")
    parser.add_argument("--dicom-root", required=True, help="Folder containing RSNA .dcm files")
    parser.add_argument("--labels", required=True, help="RSNA train.csv")
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--max-per-class", type=int, default=700, help="Keep this at <= 700 for a small project")
    args = parser.parse_args()
    labels = pd.read_csv(args.labels, usecols=["SOPInstanceUID", "pe_present_on_image"])
    labels = labels.dropna().drop_duplicates("SOPInstanceUID")
    labels["label"] = labels["pe_present_on_image"].astype(int)
    index = {p.stem: p for p in Path(args.dicom_root).rglob("*.dcm")}
    output = Path(args.output); images = output / "images"; images.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in (0, 1):
        candidates = labels[labels.label == label].sample(frac=1, random_state=42 + label)
        count = 0
        for row in candidates.itertuples(index=False):
            path = index.get(row.SOPInstanceUID)
            if path is None:
                continue
            try:
                image = resize_image(load_image(path).image)
                name = f"rsna_{label}_{count:04d}.png"
                Image.fromarray((image * 255).astype("uint8")).save(images / name, optimize=True)
                rows.append({"image_path": f"images/{name}", "label": label, "study_id": "RSNA"})
                count += 1
                if count >= args.max_per_class:
                    break
            except Exception as exc:
                print(f"Skipped {path.name}: {exc}")
        print(f"Class {label}: prepared {count} images")
    pd.DataFrame(rows).to_csv(output / "manifest.csv", index=False)
    print(f"Wrote {len(rows)} image records to {output / 'manifest.csv'}")


if __name__ == "__main__":
    main()
