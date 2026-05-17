from sklearn.cluster import KMeans
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

k_range = range(2,22)

ss_arr = []
sil_arr = []

for k in k_range:
    k_model = KMeans(n_clusters=k,random_state=22)
    k_model.fit(x_scaled)
    ss_arr.append(k_model.inertia_)
    sil_score = silhouette_score(x_scaled,k_model.labels_)
    sil_arr.append(sil_score)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex=True)
ax1.plot(k_range, ss_arr, marker='o'); ax1.set_ylabel('inertia')
ax2.plot(k_range, sil_arr, marker='x'); ax2.set_ylabel('silhouette'); ax2.set_ylim(-1,1)
ax2.set_xlabel('k')
plt.show()

