"""Assignment 2: binary spam classification using KNN and SVM."""
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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parent
DATA, OUTPUTS = ROOT / "data", ROOT / "outputs"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/balaka18/email-spam-classification-dataset-csv"


def download_dataset() -> None:
    DATA.mkdir(exist_ok=True); archive = DATA / "emails.zip"
    print("Downloading the public Kaggle dataset..."); urllib.request.urlretrieve(KAGGLE_URL, archive)
    with zipfile.ZipFile(archive) as zf: zf.extractall(DATA)
    archive.unlink(missing_ok=True)


def csv_path(given: str | None) -> Path:
    if given: return Path(given)
    files = list(DATA.glob("*.csv"))
    if not files: raise FileNotFoundError("No CSV found. Put emails.csv in data/ or run with --download.")
    return files[0]


def metric_row(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    return {"Model": name, "Accuracy": accuracy, "Precision": precision_score(y_true, y_pred, zero_division=0), "Recall": recall_score(y_true, y_pred, zero_division=0), "F1 Score": f1_score(y_true, y_pred, zero_division=0), "Specificity": tn / (tn + fp) if tn + fp else 0, "Error Rate": 1 - accuracy}, cm


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data"); parser.add_argument("--download", action="store_true"); args = parser.parse_args()
    if args.download and not list(DATA.glob("*.csv")): download_dataset()
    OUTPUTS.mkdir(exist_ok=True); df = pd.read_csv(csv_path(args.data))
    label_col = next((c for c in ["Prediction", "label", "Label", "spam", "Spam"] if c in df.columns), df.columns[-1])
    y = pd.to_numeric(df[label_col], errors="raise").astype(int)
    X = df.drop(columns=[label_col]).select_dtypes(include="number")
    if X.empty: raise ValueError("No numeric word-frequency columns found.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    models = {
        # Selecting the most informative terms prevents the 3,000-dimensional
        # vocabulary from overwhelming distance-based KNN.
        "K-Nearest Neighbors (k=5)": Pipeline([("imputer", SimpleImputer()), ("select", SelectKBest(chi2, k=min(300, X.shape[1]))), ("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5, n_jobs=-1))]),
        # Hinge loss is a linear SVM objective and is substantially more efficient
        # than kernel SVM for this 3,000-term email dataset.
        "Linear SVM": Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler()), ("model", SGDClassifier(loss="hinge", alpha=0.0001, max_iter=2000, tol=1e-3, random_state=42))]),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train); pred = model.predict(X_test); row, cm = metric_row(name, y_test, pred); rows.append(row)
        plt.figure(figsize=(5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Spam", "Spam"], yticklabels=["Not Spam", "Spam"])
        plt.title(f"Confusion Matrix - {name}"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(OUTPUTS / ("knn_confusion_matrix.png" if name.startswith("K-") else "svm_confusion_matrix.png"), dpi=180); plt.close()
    report = pd.DataFrame(rows).round(4); report.to_csv(OUTPUTS / "metrics.csv", index=False)
    result = "# Assignment 2 Result\n\nComparative performance on the held-out test set:\n\n| Model | Accuracy | Precision | Recall | F1 Score | Specificity | Error Rate |\n|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1 Score']:.4f} | {r['Specificity']:.4f} | {r['Error Rate']:.4f} |" for _, r in report.iterrows())
    best = report.sort_values("F1 Score", ascending=False).iloc[0]
    result += f"\n\n**{best['Model']}** achieved the highest F1 score ({best['F1 Score']:.4f}). Insert the two confusion-matrix images as screenshots.\n"
    (OUTPUTS / "result.md").write_text(result, encoding="utf-8"); print(result)


if __name__ == "__main__": main()
