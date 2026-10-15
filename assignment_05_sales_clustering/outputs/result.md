# Assignment 5 Result

K-Means and agglomerative hierarchical clustering were performed on 2,823 sales records using: QUANTITYORDERED, PRICEEACH, SALES, MSRP, MONTH_ID, YEAR_ID. The elbow graph was generated for k = 2 to 10. The selected number of clusters was **2**, because it gave the highest silhouette score (**0.2426**) among the tested choices.

| Evaluation | Value |
|---|---:|
| Selected clusters (k) | 2 |
| K-Means WCSS | 12385.66 |
| K-Means silhouette score | 0.2426 |
| Hierarchical silhouette score | 0.1797 |

Use `elbow_method.png`, `cluster_visualization.png`, and `dendrogram.png` as screenshots. The average numeric characteristics of each K-Means segment are in `cluster_profile.csv`.
