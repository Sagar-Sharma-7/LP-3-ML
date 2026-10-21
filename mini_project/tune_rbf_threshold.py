"""Inspect a properly held-out patient fold to choose a deployment threshold from training folds."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pe_screening import extract_features, load_image, load_manifest

frame = load_manifest("data/prepared/manifest.csv")
x = np.stack([extract_features(load_image(path).image) for path in frame.resolved_path])
y = frame.label.to_numpy(); groups = frame.study_id.astype(str).to_numpy()
outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_i, test_i = next(outer.split(x, y, groups))
def model(): return Pipeline([("scale", StandardScaler()), ("model", SVC(C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=42))])
# Threshold selection uses *only outer-training patients*.
oof = np.zeros(len(train_i))
inner_groups = groups[train_i]
inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=17)
for in_train, in_val in inner.split(x[train_i], y[train_i], inner_groups):
    m=model(); m.fit(x[train_i][in_train], y[train_i][in_train]); oof[in_val]=m.predict_proba(x[train_i][in_val])[:,1]
thresholds=np.arange(0.05,0.96,0.02)
threshold=max(thresholds, key=lambda t:f1_score(y[train_i],oof>=t,zero_division=0))
m=model(); m.fit(x[train_i],y[train_i]); p=m.predict_proba(x[test_i])[:,1]; pred=p>=threshold
print("Threshold",round(float(threshold),2)); print("Test patients", sorted(set(groups[test_i])))
print({"accuracy":round(accuracy_score(y[test_i],pred),3),"precision":round(precision_score(y[test_i],pred,zero_division=0),3),"recall":round(recall_score(y[test_i],pred,zero_division=0),3),"f1":round(f1_score(y[test_i],pred,zero_division=0),3),"roc_auc":round(roc_auc_score(y[test_i],p),3)})
