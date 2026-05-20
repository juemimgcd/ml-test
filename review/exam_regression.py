from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



data = pd.read_csv(r"D:\machineL\ml-test\review\test_data\regression_samples.csv")

X = data.iloc[:,0]
y = data.iloc[:,1]

degrees = [1, 3, 7, 9]

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

mse_dict = {}


# ----------------------------
# 3. 对每个次数进行多项式拟合和绘图
# ----------------------------
for i, degree in enumerate(degrees):
    # 使用 np.polyfit 进行拟合，得到多项式系数
    coeffs = np.polyfit(X, y, deg=degree)

    # 使用 np.poly1d 创建多项式函数
    poly_func = np.poly1d(coeffs)

    # 计算拟合值
    y_pred = poly_func(X)
    mse_dict[degree] = mean_squared_error(y, y_pred)
    # 绘图
    ax = axes[i]
    ax.scatter(X, y, alpha=0.6, label='Sample Data', color='lightblue', edgecolors='black')
    ax.plot(X, y_pred, color='red', linewidth=2, label=f'Degree {degree} Polyfit')
    ax.set_title(f'Polynomial Regression (Degree = {degree})', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


for degree in degrees:
    coeffs = np.polyfit(X, y, deg=degree)
    print(f"Degree {degree}: {coeffs},{mse_dict[degree]}")
















