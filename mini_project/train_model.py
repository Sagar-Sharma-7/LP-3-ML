"""Train the CPU-friendly classifier from a prepared manifest."""
from pathlib import Path
import argparse
from pe_screening import save_model, train_from_manifest, json_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PE screening academic prototype.")
    parser.add_argument("--manifest", default="data/prepared/manifest.csv")
    parser.add_argument("--output", default="models/pe_screening_model.joblib")
    args = parser.parse_args()
    artifact, metrics = train_from_manifest(args.manifest)
    save_model(artifact, args.output)
    print(f"Saved model to {Path(args.output)}")
    print(json_metrics(metrics))


if __name__ == "__main__":
    main()
