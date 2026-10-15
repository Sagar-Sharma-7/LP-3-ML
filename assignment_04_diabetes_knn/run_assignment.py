"""Assignment 4: diabetes classification using K-Nearest Neighbors."""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA, OUTPUTS = ROOT / "data", ROOT / "outputs"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/abdallamahgoub/diabetes"


def download_dataset():
    DATA.mkdir(exist_ok=True); archive = DATA / "diabetes.zip"; print("Downloading the public Kaggle dataset...")
    urllib.request.urlretrieve(KAGGLE_URL, archive)
    with zipfile.ZipFile(archive) as zf: zf.extractall(DATA)
    archive.unlink(missing_ok=True)


def get_csv(given):
    if given: return Path(given)
    files = list(DATA.glob("*.csv"))
    if not files: raise FileNotFoundError("No CSV found. Put diabetes.csv in data/ or run with --download.")
    return files[0]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data"); parser.add_argument("--download", action="store_true"); args = parser.parse_args()
    if args.download and not list(DATA.glob("*.csv")): download_dataset()
    OUTPUTS.mkdir(exist_ok=True); df = pd.read_csv(get_csv(args.data))
    target = "Outcome" if "Outcome" in df.columns else df.columns[-1]
    X, y = df.drop(columns=[target]).copy(), df[target].astype(int)
    zero_as_missing = [c for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"] if c in X]
    X[zero_as_missing] = X[zero_as_missing].replace(0, float("nan"))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    scores = []
    for k in range(3, 22, 2):
        pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
        scores.append((k, cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()))
    best_k = max(scores, key=lambda pair: pair[1])[0]
    model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=best_k))])
    model.fit(X_train, y_train); pred = model.predict(X_test); cm = confusion_matrix(y_test, pred, labels=[1, 0])
    tp, fn, fp, tn = cm.ravel(); accuracy = accuracy_score(y_test, pred)
    metrics = {"Accuracy": accuracy, "Error Rate": 1-accuracy, "Precision": precision_score(y_test, pred, zero_division=0), "Recall": recall_score(y_test, pred, zero_division=0), "F1 Score": f1_score(y_test, pred, zero_division=0)}
    pd.DataFrame([metrics]).round(4).to_csv(OUTPUTS / "metrics.csv", index=False)
    plt.figure(figsize=(7, 4)); plt.plot([x[0] for x in scores], [x[1] for x in scores], marker="o"); plt.axvline(best_k, color="red", linestyle="--", label=f"Selected k={best_k}")
    plt.xlabel("Number of neighbors (k)"); plt.ylabel("5-fold validation accuracy"); plt.title("KNN Model Selection"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(OUTPUTS / "k_selection_accuracy.png", dpi=180); plt.close()
    plt.figure(figsize=(5, 4)); sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
    plt.title("KNN Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(OUTPUTS / "confusion_matrix.png", dpi=180); plt.close()
    result = f"""# Assignment 4 Result

After replacing medically impossible zero values with missing values, median imputation and feature normalization were applied. The best k from 5-fold validation was **{best_k}**.

| Metric | Value |
|---|---:|
| Accuracy | {metrics['Accuracy']:.4f} |
| Error Rate | {metrics['Error Rate']:.4f} |
| Precision | {metrics['Precision']:.4f} |
| Recall | {metrics['Recall']:.4f} |
| F1-Score | {metrics['F1 Score']:.4f} |

Confusion matrix: TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}. Insert `confusion_matrix.png` and `k_selection_accuracy.png` as screenshots (the latter is the KNN training/model-selection curve).
"""
    (OUTPUTS / "result.md").write_text(result, encoding="utf-8"); print(result)


if __name__ == "__main__": main()
