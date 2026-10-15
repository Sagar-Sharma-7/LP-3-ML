"""Assignment 1: Uber fare prediction with Linear Regression and Random Forest."""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA, OUTPUTS = ROOT / "data", ROOT / "outputs"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/yasserh/uber-fares-dataset"


def download_dataset() -> None:
    DATA.mkdir(exist_ok=True)
    archive = DATA / "uber_fares.zip"
    print("Downloading the public Kaggle dataset...")
    urllib.request.urlretrieve(KAGGLE_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA)
    archive.unlink(missing_ok=True)


def locate_csv(user_path: str | None) -> Path:
    if user_path:
        return Path(user_path)
    files = list(DATA.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No CSV found. Put the Uber CSV in data/ or run with --download.")
    return files[0]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to the Uber CSV")
    parser.add_argument("--download", action="store_true", help="Download from Kaggle")
    args = parser.parse_args()
    if args.download and not list(DATA.glob("*.csv")):
        download_dataset()
    csv_path = locate_csv(args.data)
    OUTPUTS.mkdir(exist_ok=True)
    raw = pd.read_csv(csv_path, nrows=100_000)
    required = {"fare_amount", "pickup_datetime", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = raw.dropna(subset=required).copy()
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df = df[dt.notna()].copy(); dt = dt[dt.notna()]
    valid = (
        df.fare_amount.between(2.5, 500) & df.passenger_count.between(1, 6)
        & df.pickup_latitude.between(-90, 90) & df.dropoff_latitude.between(-90, 90)
        & df.pickup_longitude.between(-180, 180) & df.dropoff_longitude.between(-180, 180)
    )
    df, dt = df[valid].copy(), dt[valid]
    df["distance_km"] = haversine(df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude)
    df = df[df.distance_km.between(0.05, 100)].copy(); dt = dt.loc[df.index]
    df["hour"], df["day_of_week"], df["month"] = dt.dt.hour, dt.dt.dayofweek, dt.dt.month
    features = ["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "distance_km", "hour", "day_of_week", "month"]
    q1, q3 = df.fare_amount.quantile([0.25, 0.75]); iqr = q3 - q1
    outliers = ((df.fare_amount < q1 - 1.5 * iqr) | (df.fare_amount > q3 + 1.5 * iqr)).sum()
    corr = df[features + ["fare_amount"]].corr(numeric_only=True)
    plt.figure(figsize=(10, 8)); sns.heatmap(corr, cmap="coolwarm", center=0); plt.tight_layout(); plt.savefig(OUTPUTS / "correlation_heatmap.png", dpi=180); plt.close()
    X_train, X_test, y_train, y_test = train_test_split(df[features], df.fare_amount, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": Pipeline([("imputer", SimpleImputer()), ("model", LinearRegression())]),
        "Random Forest Regression": Pipeline([("imputer", SimpleImputer()), ("model", RandomForestRegressor(n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1))]),
    }
    rows, predictions = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train); pred = model.predict(X_test); predictions[name] = pred
        rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred), "RMSE": mean_squared_error(y_test, pred) ** 0.5, "R2 Score": r2_score(y_test, pred)})
    report = pd.DataFrame(rows).round(4); report.to_csv(OUTPUTS / "metrics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, pred) in zip(axes, predictions.items()):
        ax.scatter(y_test, pred, s=8, alpha=.35); lo, hi = y_test.min(), y_test.max(); ax.plot([lo, hi], [lo, hi], "r--")
        ax.set(title=name, xlabel="Actual fare ($)", ylabel="Predicted fare ($)")
    fig.tight_layout(); fig.savefig(OUTPUTS / "actual_vs_predicted.png", dpi=180); plt.close(fig)
    best = report.sort_values("R2 Score", ascending=False).iloc[0]
    result = f"""# Assignment 1 Result

Dataset rows used after preprocessing: {len(df):,}. IQR method identified {outliers:,} fare outliers (reported for analysis; valid data were retained for a fair model comparison).

| Model | MAE | RMSE | R2 Score |
|---|---:|---:|---:|
""" + "\n".join(f"| {r['Model']} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['R2 Score']:.4f} |" for _, r in report.iterrows()) + f"""

The best model was **{best['Model']}** with R2 = {best['R2 Score']:.4f} and RMSE = {best['RMSE']:.4f}. Use `correlation_heatmap.png` and `actual_vs_predicted.png` as screenshots.
"""
    (OUTPUTS / "result.md").write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
