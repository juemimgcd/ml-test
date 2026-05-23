from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


data = pd.read_csv(r"E:\python_files\ml-test\review\test_data\synthetic_data.csv")

X = data.iloc[:,0]
y = data.iloc[:,1]

degree = [1,5,15]
fig,axes = plt.subplots(2,2,figsize=(10,10))
ax = axes.flatten()
mse_array = []

model_score = {}
for i,degree in enumerate(degree):
    coeffs =np.polyfit(X,y,deg=degree)
    func = np.poly1d(coeffs)
    y_pred = func(X)


    mse_array.append(mean_squared_error(y_true=y,y_pred=y_pred))
    model_score[degree] = mse_array[i]

    ax[i].scatter(X,y,marker='*',c='r')
    ax[i].plot(X,y_pred,label="Predicted")
    ax[i].set_title(f"Degree = {str(degree)},SSE = {model_score[degree]}")
    ax[i].legend()
    ax[i].set_xlabel("X")
    ax[i].set_ylabel("Y")

plt.tight_layout()
plt.show()






