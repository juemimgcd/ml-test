from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"D:\machineL\ml-test\review\test_data\yeast.csv")
X = data.iloc[:,1:-1]
y = data.iloc[:,-1]

scaler = StandardScaler()
X = scaler.fit_transform(X)
fig,axes = plt.subplots(1,3,figsize=(20,8))
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)


min_pts = [3,5,10]

for index,k in enumerate(min_pts):
    model = DBSCAN(min_samples=k,eps=0.5)
    model.fit(X_pca)
    labels = model.labels_

    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            name = "noise"
            color = "black"
        else:
            name = "yeast"
            color = plt.cm.tab10(label % 10)
        data = X_pca[labels == label]
        axes[index].scatter(data[:,0],data[:,1])
        axes[index].set_title(f"min_pts={k}")

plt.tight_layout()
plt.show()





























