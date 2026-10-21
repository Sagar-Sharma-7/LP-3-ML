"""Shared utilities for the CPU-only PE screening academic prototype.

This is intentionally a research/teaching utility. It does not diagnose PE.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import io
import json

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

IMAGE_SIZE = 224
MODEL_VERSION = "1.0"


@dataclass
class LoadedImage:
    image: np.ndarray
    metadata: dict


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low, high = np.percentile(values[np.isfinite(values)], [1, 99]) if np.isfinite(values).any() else (0, 1)
    if high <= low:
        high = low + 1.0
    return np.clip((values - low) / (high - low), 0, 1)


def load_image(source: str | Path | bytes, filename: str | None = None) -> LoadedImage:
    """Load a PNG/JPEG or a single DICOM slice and return a 0..1 greyscale image."""
    path_name = str(source) if isinstance(source, (str, Path)) else (filename or "uploaded")
    suffix = Path(path_name).suffix.lower()
    metadata: dict = {"source": Path(path_name).name, "format": suffix.lstrip(".").upper() or "UNKNOWN"}
    if suffix == ".dcm":
        import pydicom
        ds = pydicom.dcmread(source, force=True)
        pixels = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixels = pixels * slope + intercept
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            pixels = -pixels
        image = _normalise(pixels)
        metadata.update({
            "format": "DICOM", "rows": int(getattr(ds, "Rows", pixels.shape[0])),
            "columns": int(getattr(ds, "Columns", pixels.shape[1])),
            "modality": str(getattr(ds, "Modality", "Unknown")),
            "study_uid": str(getattr(ds, "StudyInstanceUID", "Not available"))[-18:],
        })
    else:
        image = np.asarray(Image.open(source).convert("L"), dtype=np.float32) / 255.0
        metadata.update({"rows": int(image.shape[0]), "columns": int(image.shape[1]), "modality": "Image"})
    return LoadedImage(image=np.clip(image, 0, 1), metadata=metadata)


def resize_image(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(uint8).resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def extract_features(image: np.ndarray) -> np.ndarray:
    """CPU-friendly intensity, gradient, histogram and downsampled image features."""
    image = resize_image(image)
    gx, gy = np.gradient(image)
    grad = np.sqrt(gx * gx + gy * gy)
    hist = np.histogram(image, bins=24, range=(0, 1), density=True)[0]
    g_hist = np.histogram(grad, bins=16, range=(0, max(float(grad.max()), 0.01)), density=True)[0]
    # 32x32 image representation helps the linear model recognise broad patterns.
    small = np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    stats = np.array([image.mean(), image.std(), np.percentile(image, 10), np.percentile(image, 50), np.percentile(image, 90), grad.mean(), grad.std()], dtype=np.float32)
    return np.concatenate([stats, hist.astype(np.float32), g_hist.astype(np.float32), small.ravel()])


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(manifest_path)
    required = {"image_path", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("Manifest must include image_path and label columns.")
    frame["label"] = frame["label"].astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 (PE absent) or 1 (PE present).")
    frame["resolved_path"] = frame["image_path"].map(lambda p: str((manifest_path.parent / p).resolve()) if not Path(p).is_absolute() else p)
    missing = [p for p in frame["resolved_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Manifest references missing image: {missing[0]}")
    return frame


def train_from_manifest(manifest_path: str | Path, test_size: float = 0.2, random_state: int = 42) -> tuple[dict, dict]:
    frame = load_manifest(manifest_path)
    if len(frame) < 20 or frame["label"].nunique() != 2:
        raise ValueError("Use at least 20 images and both classes to train a model.")
    features = np.stack([extract_features(load_image(p).image) for p in frame["resolved_path"]])
    labels = frame["label"].to_numpy()
    # A patient/study must never appear in both train and test data. Without this,
    # adjacent near-identical CT slices would make performance look misleadingly high.
    pipeline = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear", random_state=random_state))])
    if "study_id" in frame.columns and frame["study_id"].nunique() >= 4:
        groups = frame["study_id"].astype(str).to_numpy()
        folds = min(5, frame["study_id"].nunique())
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
        probabilities = np.zeros(len(frame), dtype=np.float32)
        for train_indices, test_indices in splitter.split(features, labels, groups):
            fold_model = clone(pipeline)
            fold_model.fit(features[train_indices], labels[train_indices])
            probabilities[test_indices] = fold_model.predict_proba(features[test_indices])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        y_test = labels
        held_out_studies = int(frame["study_id"].nunique())
        validation_scheme = f"{folds}-fold patient-level cross-validation"
        # Fit one final model on every available training slice for app inference.
        pipeline.fit(features, labels)
    else:
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        held_out_studies = 0
        validation_scheme = "stratified hold-out split"
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)), "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)), "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)), "samples": int(len(frame)), "test_samples": int(len(y_test)),
        "positive_samples": int(labels.sum()), "held_out_studies": held_out_studies, "validation_scheme": validation_scheme, "model_version": MODEL_VERSION,
    }
    artifact = {"pipeline": pipeline, "metrics": metrics, "image_size": IMAGE_SIZE, "feature_count": int(features.shape[1])}
    return artifact, metrics


def save_model(artifact: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)


def predict_image(artifact: dict, image: np.ndarray) -> float:
    features = extract_features(image).reshape(1, -1)
    return float(artifact["pipeline"].predict_proba(features)[0, 1])


def simple_heatmap(artifact: dict, image: np.ndarray, grid: int = 8) -> np.ndarray:
    """Occlusion sensitivity map: explanatory aid, not a medical localization."""
    base = predict_image(artifact, image)
    image = resize_image(image)
    output = np.zeros((grid, grid), dtype=np.float32)
    block = IMAGE_SIZE // grid
    fill = float(np.median(image))
    for y in range(grid):
        for x in range(grid):
            altered = image.copy()
            altered[y * block:(y + 1) * block, x * block:(x + 1) * block] = fill
            output[y, x] = max(0.0, base - predict_image(artifact, altered))
    enlarged = Image.fromarray((output / (output.max() or 1) * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(enlarged, dtype=np.float32) / 255.0


def create_report(study_name: str, records: list[dict], model_metrics: dict) -> tuple[pd.DataFrame, bytes]:
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No predictions to report.")
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfgen import canvas
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=A4)
    width, height = A4
    styles = getSampleStyleSheet()
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(42, height - 48, "PE Screening Prototype – Academic Report")
    pdf.setFont("Helvetica", 9)
    y = height - 72
    lines = [
        f"Study / input: {study_name}", f"Analysed images: {len(frame)}", f"Highest slice score: {frame['pe_probability'].max():.1%}",
        f"Mean slice score: {frame['pe_probability'].mean():.1%}", f"Training model F1: {model_metrics.get('f1', 0):.1%}",
        "IMPORTANT: This research prototype is not a medical device and must not be used to diagnose or rule out pulmonary embolism.",
    ]
    for line in lines:
        for segment in [line[i:i + 105] for i in range(0, len(line), 105)]:
            pdf.drawString(42, y, segment); y -= 14
    y -= 8
    pdf.setFont("Helvetica-Bold", 10); pdf.drawString(42, y, "Top-scoring images"); y -= 14
    pdf.setFont("Helvetica", 9)
    for _, row in frame.sort_values("pe_probability", ascending=False).head(12).iterrows():
        pdf.drawString(42, y, f"{row['image'][:70]}  —  score: {row['pe_probability']:.1%}"); y -= 13
    pdf.save()
    return frame, output.getvalue()


def json_metrics(metrics: dict) -> str:
    return json.dumps(metrics, indent=2)
