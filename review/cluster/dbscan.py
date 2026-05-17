from IPython.core.pylabtools import figsize
from matplotlib.pyplot import figure
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"E:\python_test\dig\review\test_data\yeast.csv")
x = data.iloc[:,1:-1]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

def db(n):
    dbscan = DBSCAN(eps=0.8, min_samples=n)
    labels = dbscan.fit_predict(x_scaled)

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)


    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)

    n_noise = np.sum(labels==-1)

    plt.figure(figsize=(10,8))

    unique_labels = np.unique(labels)

    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            plt.scatter(
                x_pca[mask, 0], x_pca[mask, 1],
                c="black", s=18, alpha=0.6, label="Noise (-1)"
            )
        else:
            plt.scatter(
                x_pca[mask, 0], x_pca[mask, 1],
                s=22, alpha=0.75, label=f"Cluster {lab}"
            )

    plt.title(f"DBSCAN Clustering (clusters={n_clusters}, noise={n_noise})")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend()


p1 = db(5)
p2 = db(3)
p3 = db(10)
plt.show()



