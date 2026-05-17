import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

x,_ = make_moons(
    n_samples=400,
    noise=0.03,
    random_state=42
)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

eps = 0.25
min_ps = [5, 6, 7, 9]

result = []
for number in min_ps:
    model = DBSCAN(eps=eps, min_samples=number)
    labels = model.fit_predict(x_scaled)

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = np.sum(labels == -1)

    if n_clusters >= 2:
        mask = labels != -1
        score = silhouette_score(x[mask], labels[mask])

    else:
        score = np.nan
    result.append((number, n_clusters, n_noise, score))


best = max(result, key=lambda t: t[3])
cluster = best[0]

b_model = DBSCAN(eps=eps,min_samples=cluster)
b_labels = b_model.fit_predict(x_scaled)

unique_labels = set(b_labels)

for lab in unique_labels:
    if lab == -1:
        color = "black"
        name = "noise"
    else:
        name = f"{lab}"
        color = plt.cm.tab10(lab % 10)

    data = x_scaled[b_labels == lab]

    plt.scatter(data[:,0],data[:,1],c=color,label=name)


plt.show()












































