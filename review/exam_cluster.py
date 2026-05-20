from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,DBSCAN
import pandas as pd
import numpy as np

data = pd.read_csv(r"D:\machineL\ml-test\review\test_data\yeast.csv")
print(data.head())

X = data.iloc[:,1:-1]
y = data.iloc[:,-1]
X_scaled = StandardScaler().fit_transform(X)

# sse_list = []
# for k in range(2,12):
#     model = KMeans(n_clusters=k,random_state=42)
#     model.fit(X_scaled)
#     sse_list.append(model.inertia_)
#
# plt.figure(figsize=(10,10))
# plt.plot(range(2,12),sse_list,marker='o')
# plt.show()
#
# for k, sse in zip(range(2,12), sse_list):
#     print(f"K={k}, SSE={sse:.2f}")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
min_pts = [3,5,10]

fig,axes = plt.subplots(1,3,figsize=(20,5))

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



























