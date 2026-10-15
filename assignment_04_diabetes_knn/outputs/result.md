# Assignment 4 Result

After replacing medically impossible zero values with missing values, median imputation and feature normalization were applied. The best k from 5-fold validation was **15**.

| Metric | Value |
|---|---:|
| Accuracy | 0.7403 |
| Error Rate | 0.2597 |
| Precision | 0.6522 |
| Recall | 0.5556 |
| F1-Score | 0.6000 |

Confusion matrix: TN = 84, FP = 16, FN = 24, TP = 30. Insert `confusion_matrix.png` and `k_selection_accuracy.png` as screenshots (the latter is the KNN training/model-selection curve).
