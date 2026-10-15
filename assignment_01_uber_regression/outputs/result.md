# Assignment 1 Result

Dataset rows used after preprocessing: 96,181. IQR method identified 8,167 fare outliers (reported for analysis; valid data were retained for a fair model comparison).

| Model | MAE | RMSE | R2 Score |
|---|---:|---:|---:|
| Linear Regression | 2.2701 | 4.5930 | 0.7559 |
| Random Forest Regression | 1.9327 | 3.8736 | 0.8264 |

The best model was **Random Forest Regression** with R2 = 0.8264 and RMSE = 3.8736. Use `correlation_heatmap.png` and `actual_vs_predicted.png` as screenshots.
