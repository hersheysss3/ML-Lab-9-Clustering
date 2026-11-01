"""ML Lab 9 - K-Means and Hierarchical clustering on the Wholesale Customers dataset (Wholesale customers data.csv)."""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram

# Seaborn style
sns.set_style("whitegrid")

# Matplotlib settings
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 12

df = pd.read_csv("Wholesale customers data.csv")

print(df.head())

print(df.tail())

print(df.columns)

print("Dataset shape:", df.shape)

print(df.dtypes)

print(df.info())

print(df.describe())

print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

X = df.drop(columns=["Channel", "Region"])
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns,
    index=X.index
)

print(X_scaled_df.head())

print(X_scaled_df.mean().round(3))

wcss=[]
k_values=range(1,11)
for k in k_values:
    kmeans=KMeans(n_clusters=k,random_state=42,n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker="o", linestyle="-")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method for Optimal K")
plt.xticks(k_values)
plt.grid(True)
plt.show()
    

silhouette_values = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_values.append(score)

silhouette_table = pd.DataFrame({
    "Number of Clusters": list(range(2, 11)),
    "Silhouette Score": silhouette_values
})

print(silhouette_table.round(4))

suggested_k = int(
    silhouette_table.loc[
        silhouette_table["Silhouette Score"].idxmax(),
        "Number of Clusters"
    ]
)

print("Suggested number of clusters:", suggested_k)

K = suggested_k

kmeans_model = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans_model.fit_predict(X_scaled)

df_kmeans = df.copy()

df_kmeans["kMeans_cluster"] = kmeans_labels

print("K-Means clustering completed successfully!")
print("\nCluster distribution:")
print(df_kmeans["kMeans_cluster"].value_counts().sort_index())

print("\nFirst five rows:")
print(df_kmeans.head())

plt.figure(figsize=(16, 7))

Z = linkage(X_scaled, method="ward", metric="euclidean")

dendrogram(
    Z,
    leaf_rotation=90,
    leaf_font_size=6
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Cluster Index")
plt.ylabel("Ward Linkage Distance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    show_leaf_counts=True,
    leaf_rotation=45
)

plt.title("Truncated Dendogram- last 30 Merged groups")
plt.xlabel("Cluster or Group")
plt.ylabel("Ward LinkageDistance")
plt.tight_layout()
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(
    "Variance explained by each PCA component:",
    pca.explained_variance_ratio_.round(4)
)

print(
    "Total variance explained:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "KMeans_Cluster": kmeans_labels
})

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="KMeans_Cluster",
    palette="tab10",
    s=50,
    alpha=0.7
)

plt.title("K-Means Clusters Visualized Using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

def evaluate_clustering(X_data, labels, model_name):
    return {
        "Algorithm": model_name,
        "Number of Clusters": len(set(labels)),
        "Silhouette Score": silhouette_score(X_data, labels)
    }

hierarchical_model = AgglomerativeClustering(
    n_clusters=suggested_k,
    linkage="ward"
)
hierarchical_labels = hierarchical_model.fit_predict(X_scaled)

comparison = pd.DataFrame([
    evaluate_clustering(X_scaled, kmeans_labels, "K-Means"),
    evaluate_clustering(X_scaled, hierarchical_labels, "Hierarchical")
])

print(comparison.round(4))

# Compare cluster membership sizes
cluster_sizes = pd.DataFrame({
    "K-Means": pd.Series(kmeans_labels).value_counts().sort_index(),
    "Hierarchical": pd.Series(hierarchical_labels).value_counts().sort_index()
}).fillna(0).astype(int)

print("Cluster sizes:")
print(cluster_sizes)

# Calculate average spending for each K-Means cluster
features = list(X.columns)
kmeans_profile = df[features].copy()
kmeans_profile["Cluster"] = kmeans_labels

kmeans_means = kmeans_profile.groupby("Cluster")[features].mean().round(2)

print("Average spending profile of K-Means clusters:")
print(kmeans_means)
