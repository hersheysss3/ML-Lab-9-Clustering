# ML Lab-9 : K-Means and Hierarchical Clustering on the Wholesale Customers Dataset

**Name:** Harshit Widhwani &nbsp;|&nbsp; **Section:** B &nbsp;|&nbsp; **Roll No:** 49 &nbsp;|&nbsp; **Batch:** B4

Ramdeobaba University, Nagpur | Department of Artificial Intelligence and Machine Learning

---

## Aim

To implement K-Means Clustering and Hierarchical Clustering algorithms using suitable dataset.

## Dataset

`Wholesale customers data.csv`, the UCI Wholesale Customers dataset (440 clients of a wholesale distributor, annual spending on 6 product categories, plus `Channel` and `Region`).

## Files in this repository

| File | Description |
|---|---|
| [`b-b4-49_prac9_ml.ipynb`](b-b4-49_prac9_ml.ipynb) | **Main notebook with every step and all outputs already executed.** Opens directly in the browser. |
| `kmeans_hierarchical_clustering.py` | The same pipeline as a plain Python script. |
| `Wholesale customers data.csv` | The dataset used. |

## Steps applied

1. Loaded `Wholesale customers data.csv` and inspected it with `head()`, `tail()`, `shape`, `dtypes`, `info()`, `describe()`, null and duplicate checks
2. Dropped `Channel` and `Region` and standardised the six spending features with `StandardScaler`
3. Applied the **Elbow method** (WCSS for K = 1 to 10) and the **Silhouette score** (K = 2 to 10) to choose K = 3
4. Fitted **K-Means** with K = 3 and added the cluster labels to the data
5. Built the Ward linkage matrix and drew the full and truncated **dendrograms**
6. Visualised the K-Means clusters in 2-D with **PCA** (72.46% of the variance explained)
7. Fitted **Agglomerative (hierarchical) clustering** with Ward linkage and compared it with K-Means
8. Compared the cluster sizes and the average spending profile of each K-Means cluster

## Results

| Algorithm | Clusters | Silhouette score | Cluster sizes |
|---|---|---|---|
| K-Means | 3 | 0.5483 | 45, 393, 2 |
| Hierarchical (Ward) | 3 | 0.2646 | 6, 281, 153 |

## How to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy notebook
```

Open the notebook (or run the `.py` script) from inside this folder so the dataset path resolves.
