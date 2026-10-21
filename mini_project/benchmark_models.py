"""Compare CPU-friendly classifiers using patient-level cross validation.

This script is for honest model selection. It never mixes slices from the same
patient across a train/test fold.
"""
from pathlib import Path
import argparse
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from pe_screening import extract_features, load_image, load_manifest


def models():
    return {
        "logistic_regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear", random_state=42))]),
        "rbf_svm": Pipeline([("scale", StandardScaler()), ("model", SVC(C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=42))]),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, max_features=0.55, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=400, max_features=0.45, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/prepared/manifest.csv")
    args = parser.parse_args()
    frame = load_manifest(args.manifest)
    x = np.stack([extract_features(load_image(path).image) for path in frame.resolved_path])
    y = frame.label.to_numpy()
    groups = frame.study_id.astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"Evaluating {len(frame)} real slices from {len(set(groups))} patients across 5 patient-level folds.\n")
    for name, estimator in models().items():
        values = []
        for train_index, test_index in splitter.split(x, y, groups):
            estimator.fit(x[train_index], y[train_index])
            probability = estimator.predict_proba(x[test_index])[:, 1]
            prediction = (probability >= 0.5).astype(int)
            values.append([accuracy_score(y[test_index], prediction), precision_score(y[test_index], prediction, zero_division=0), recall_score(y[test_index], prediction, zero_division=0), f1_score(y[test_index], prediction, zero_division=0), roc_auc_score(y[test_index], probability)])
        mean = np.mean(values, axis=0)
        print(f"{name:20} accuracy={mean[0]:.3f} precision={mean[1]:.3f} recall={mean[2]:.3f} f1={mean[3]:.3f} roc_auc={mean[4]:.3f}")


if __name__ == "__main__":
    main()
