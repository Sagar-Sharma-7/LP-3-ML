"""Assignment 3: bank-customer churn prediction with a neural-network classifier."""
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA, OUTPUTS = ROOT / "data", ROOT / "outputs"
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/barelydedicated/bank-customer-churn-modeling"


def download_dataset():
    DATA.mkdir(exist_ok=True); archive = DATA / "churn.zip"; print("Downloading the public Kaggle dataset...")
    urllib.request.urlretrieve(KAGGLE_URL, archive)
    with zipfile.ZipFile(archive) as zf: zf.extractall(DATA)
    archive.unlink(missing_ok=True)


def get_csv(given):
    if given: return Path(given)
    files = list(DATA.glob("*.csv"))
    if not files: raise FileNotFoundError("No CSV found. Put Churn_Modelling.csv in data/ or run with --download.")
    return files[0]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data"); parser.add_argument("--download", action="store_true"); args = parser.parse_args()
    if args.download and not list(DATA.glob("*.csv")): download_dataset()
    OUTPUTS.mkdir(exist_ok=True); df = pd.read_csv(get_csv(args.data))
    target = "Exited" if "Exited" in df.columns else df.columns[-1]
    drop = [c for c in [target, "RowNumber", "CustomerId", "Surname"] if c in df.columns]
    X, y = df.drop(columns=drop), df[target].astype(int)
    numeric = X.select_dtypes(include="number").columns.tolist(); categorical = [c for c in X.columns if c not in numeric]
    preprocess = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
    ann = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=.0005, learning_rate_init=.001, early_stopping=True, validation_fraction=.15, n_iter_no_change=20, max_iter=350, random_state=42)
    model = Pipeline([("preprocess", preprocess), ("ann", ann)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    model.fit(X_train, y_train); pred = model.predict(X_test); cm = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    pd.DataFrame(cm, index=["Actual Stay", "Actual Churn"], columns=["Predicted Stay", "Predicted Churn"]).to_csv(OUTPUTS / "confusion_matrix.csv")
    plt.figure(figsize=(5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stay (0)", "Churn (1)"], yticklabels=["Stay (0)", "Churn (1)"])
    plt.title("ANN Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.savefig(OUTPUTS / "confusion_matrix.png", dpi=180); plt.close()
    losses = model.named_steps["ann"].loss_curve_
    plt.figure(figsize=(7, 4)); plt.plot(range(1, len(losses)+1), losses); plt.xlabel("Iteration"); plt.ylabel("Training loss"); plt.title("ANN Training Loss Curve"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(OUTPUTS / "training_loss_curve.png", dpi=180); plt.close()
    report = classification_report(y_test, pred, target_names=["Stay", "Churn"], digits=4)
    (OUTPUTS / "classification_report.txt").write_text(report, encoding="utf-8")
    result = f"""# Assignment 3 Result

The ANN achieved an accuracy of **{accuracy:.4f} ({accuracy * 100:.2f}%)** on the test set. It was trained with standardized numerical features, one-hot encoded categorical features, two hidden layers, and early stopping to reduce overfitting.

Confusion matrix (rows: actual; columns: predicted):

| | Predicted Stay (0) | Predicted Churn (1) |
|---|---:|---:|
| Actual Stay (0) | {cm[0,0]} | {cm[0,1]} |
| Actual Churn (1) | {cm[1,0]} | {cm[1,1]} |

Insert `confusion_matrix.png` and `training_loss_curve.png` as screenshots. Detailed precision, recall, and F1 score are in `classification_report.txt`.
"""
    (OUTPUTS / "result.md").write_text(result, encoding="utf-8"); print(result)


if __name__ == "__main__": main()
