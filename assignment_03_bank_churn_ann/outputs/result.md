# Assignment 3 Result

The ANN achieved an accuracy of **0.8655 (86.55%)** on the test set. It was trained with standardized numerical features, one-hot encoded categorical features, two hidden layers, and early stopping to reduce overfitting.

Confusion matrix (rows: actual; columns: predicted):

| | Predicted Stay (0) | Predicted Churn (1) |
|---|---:|---:|
| Actual Stay (0) | 1526 | 67 |
| Actual Churn (1) | 202 | 205 |

Insert `confusion_matrix.png` and `training_loss_curve.png` as screenshots. Detailed precision, recall, and F1 score are in `classification_report.txt`.
