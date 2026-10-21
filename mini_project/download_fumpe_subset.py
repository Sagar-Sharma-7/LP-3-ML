"""Download a small public FUMPE subset using Hugging Face's transfer client.

Default: 12 CTA studies plus their expert masks, usually about 1.3 GB. This is a
real, public dataset, not a demo dataset. See README for attribution and limits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

REPOSITORY = "MedOtter/FUMPE"


def download(repository_file: str, destination_root: Path) -> None:
    destination = destination_root / repository_file
    if destination.exists():
        print(f"Already present: {repository_file}")
        return
    print(f"Downloading {repository_file} through the Hugging Face transfer client…")
    hf_hub_download(
        repo_id=REPOSITORY,
        repo_type="dataset",
        filename=repository_file,
        local_dir=destination_root,
    )
    if not destination.exists() or destination.stat().st_size == 0:
        raise SystemExit(f"The transfer client did not create {destination}.")
    print(f"Saved {repository_file} ({destination.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a storage-capped FUMPE subset with CTA images and expert masks.")
    parser.add_argument("--output", default="data/raw/fumpe")
    parser.add_argument("--patients", type=int, default=12, help="1–35; 12 is recommended for this CPU-only project")
    parser.add_argument("--start", type=int, default=1, help="First patient number, e.g. 1 for PAT001")
    args = parser.parse_args()
    if not 1 <= args.start <= 35 or not 1 <= args.patients <= 35 or args.start + args.patients - 1 > 35:
        raise SystemExit("Choose a valid contiguous range within PAT001–PAT035.")
    root = Path(args.output)
    for number in range(args.start, args.start + args.patients):
        patient = f"PAT{number:03d}.nii.gz"
        download(f"images/{patient}", root)
        download(f"masks/{patient}", root)
    print("Download complete. Next run: python prepare_fumpe_subset.py --fumpe-root data/raw/fumpe")


if __name__ == "__main__":
    main()
