import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from review.cluster.dbscan2 import n_clusters

x, _ = make_moons(
    n_samples=400,
    noise=0.03,
    random_state=42
)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
result = []

for k in range(2, 22):
    model = DBSCAN(min_samples=k)
    model.fit(x_scaled)
    labels = model.labels_

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = np.sum(labels == -1)

    for lab in labels:
        if lab == -1:
            mask = labels != -1
            score = silhouette_score(x[mask],labels[mask])

        else:
            score = np.nan

        result.append((k,n_clusters,n_noise,score))


best_k = max(result,key=lambda p:p[3])

best_model = DBSCAN(min_samples=best_k[0])

best_labels = best_model.fit_predict(x_scaled)

unique_labels = set(best_labels)

for lab in unique_labels:
    if lab == -1:
        name = "noise"
        color = "black"
    else:
        name = "point"
        color = plt.cm.tab10(lab % 10)

    data = x_scaled[best_labels == lab]
    plt.scatter(data[:,0],data[:,1])

plt.show()









