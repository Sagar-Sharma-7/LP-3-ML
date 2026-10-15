"""Assignment 5: K-Means and hierarchical clustering of sample sales data."""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA, OUTPUTS = ROOT / "data", ROOT / "outputs"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/kyanyoga/sample-sales-data"


def download_dataset():
    DATA.mkdir(exist_ok=True); archive = DATA / "sales.zip"; print("Downloading the public Kaggle dataset...")
    urllib.request.urlretrieve(KAGGLE_URL, archive)
    with zipfile.ZipFile(archive) as zf: zf.extractall(DATA)
    archive.unlink(missing_ok=True)


def get_csv(given):
    if given: return Path(given)
    files = list(DATA.glob("*.csv"))
    if not files: raise FileNotFoundError("No CSV found. Put sales_data_sample.csv in data/ or run with --download.")
    return files[0]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data"); parser.add_argument("--download", action="store_true"); args = parser.parse_args()
    if args.download and not list(DATA.glob("*.csv")): download_dataset()
    OUTPUTS.mkdir(exist_ok=True); df = pd.read_csv(get_csv(args.data), encoding="latin1")
    selected = [c for c in ["QUANTITYORDERED", "PRICEEACH", "SALES", "MSRP", "MONTH_ID", "YEAR_ID"] if c in df.columns]
    if len(selected) < 2: raise ValueError("Expected numeric sales columns were not found.")
    X = df[selected].apply(pd.to_numeric, errors="coerce")
    prepared = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]).fit_transform(X)
    max_k = min(10, len(prepared) - 1); ks = list(range(2, max_k + 1)); inertias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(prepared)
        inertias.append(km.inertia_); silhouettes.append(silhouette_score(prepared, km.labels_))
    best_k = ks[int(np.argmax(silhouettes))]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit(prepared)
    hierarchy = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(prepared)
    pd.DataFrame({"k": ks, "WCSS": inertias, "Silhouette Score": silhouettes}).round(4).to_csv(OUTPUTS / "cluster_scores.csv", index=False)
    plt.figure(figsize=(7, 4)); plt.plot(ks, inertias, marker="o"); plt.axvline(best_k, color="red", linestyle="--", label=f"Chosen k={best_k}")
    plt.title("Elbow Method for K-Means"); plt.xlabel("Number of clusters (k)"); plt.ylabel("WCSS / inertia"); plt.xticks(ks); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(OUTPUTS / "elbow_method.png", dpi=180); plt.close()
    coords = PCA(n_components=2, random_state=42).fit_transform(prepared)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, labels, title in zip(axes, [kmeans.labels_, hierarchy.labels_], ["K-Means Clusters", "Hierarchical Clusters"]):
        points = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18, alpha=.75); ax.set(title=title, xlabel="PCA component 1", ylabel="PCA component 2"); fig.colorbar(points, ax=ax, label="Cluster")
    fig.tight_layout(); fig.savefig(OUTPUTS / "cluster_visualization.png", dpi=180); plt.close(fig)
    sample = prepared[:min(150, len(prepared))]
    plt.figure(figsize=(12, 5)); dendrogram(linkage(sample, method="ward"), no_labels=True, color_threshold=None); plt.title("Hierarchical Clustering Dendrogram (first 150 records)"); plt.xlabel("Sample index"); plt.ylabel("Ward linkage distance"); plt.tight_layout(); plt.savefig(OUTPUTS / "dendrogram.png", dpi=180); plt.close()
    cluster_summary = df.assign(cluster=kmeans.labels_).groupby("cluster")[selected].mean(numeric_only=True).round(2)
    cluster_summary.to_csv(OUTPUTS / "cluster_profile.csv")
    result = f"""# Assignment 5 Result

K-Means and agglomerative hierarchical clustering were performed on {len(df):,} sales records using: {', '.join(selected)}. The elbow graph was generated for k = {ks[0]} to {ks[-1]}. The selected number of clusters was **{best_k}**, because it gave the highest silhouette score (**{max(silhouettes):.4f}**) among the tested choices.

| Evaluation | Value |
|---|---:|
| Selected clusters (k) | {best_k} |
| K-Means WCSS | {kmeans.inertia_:.2f} |
| K-Means silhouette score | {silhouette_score(prepared, kmeans.labels_):.4f} |
| Hierarchical silhouette score | {silhouette_score(prepared, hierarchy.labels_):.4f} |

Use `elbow_method.png`, `cluster_visualization.png`, and `dendrogram.png` as screenshots. The average numeric characteristics of each K-Means segment are in `cluster_profile.csv`.
"""
    (OUTPUTS / "result.md").write_text(result, encoding="utf-8"); print(result)


if __name__ == "__main__": main()
