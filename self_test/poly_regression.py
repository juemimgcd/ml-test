from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"D:\machineL\ml-test\review\test_data\regression_samples.csv")

X = data.iloc[:,0]
y = data.iloc[:,1]

degree = [3,5,10]
fig,axes = plt.subplots(2,2,figsize=(10,10))
ax = axes.flatten()
sse_array = []

model_score = {}
for i,degree in enumerate(degree):
    coeffs =np.polyfit(X,y,deg=degree)
    func = np.poly1d(coeffs)
    y_pred = func(X)

    r = y_pred - y
    sse_array.append(np.sum(np.square(r)))
    model_score[degree] = sse_array[i]

    ax[i].scatter(X,y,marker='*',c='r')
    ax[i].plot(X,y_pred,label="Predicted")
    ax[i].set_title(f"Degree = {str(degree)},SSE = {model_score[degree]}")
    ax[i].legend()
    ax[i].set_xlabel("X")
    ax[i].set_ylabel("Y")

plt.tight_layout()
plt.show()






