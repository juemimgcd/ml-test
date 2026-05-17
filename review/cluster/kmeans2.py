import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

x,_ = make_blobs(
    n_samples=400,
    centers=4,
    cluster_std=0.3,
    random_state=42
)


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

result = []
for k in range(2,12):
    k_model = KMeans(n_clusters=k,random_state=42,n_init=10)
    k_model.fit(x_scaled)
    labels = k_model.labels_
    score = silhouette_score(x_scaled,labels)
    result.append((k,score))

bestK = max(result,key=lambda x:x[1])[0]

best_model = KMeans(n_clusters=bestK,random_state=42,n_init=10)
labs = best_model.fit_predict(x_scaled)

plt.scatter(x[:,0],x[:,1],c=labs)

centers = scaler.inverse_transform(best_model.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1],c="black",marker="x")




plt.show()

















