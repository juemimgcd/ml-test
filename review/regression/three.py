import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.figure(figsize=(8,8))


def runge(x):
    return 1 / (1 + 25 * x ** 2)


np.random.seed(42)
x = np.linspace(-1, 1, 120)
y_true = runge(x)
y = y_true + np.random.normal(0, 0.03, size=len(x))

X = x.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


degree = 20

models = {
    "Ridge": Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Lasso": Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.001, max_iter=20000))
    ]),
    "PolynomialRegression": Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", LinearRegression())
    ])
}


def regression_metrics(y_true, y_pred):
    r = y_pred - y_true
    SSE = np.sum(r**2)
    MSE = mean_squared_error(y_true, y_pred=y_pred)
    MAE = mean_absolute_error(y_true,y_pred)


    return r,MSE,MAE,SSE


predict = {}

for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    r,mse,mae,sse = regression_metrics(y_test,y_pred)

    predict[name] = [y_pred,r,mse,mae,sse]



plt.subplot(221)
plt.scatter(X_train,y_train,label="train")
plt.scatter(X_test,predict["Ridge"][0],label="Ridge test")
plt.legend()

plt.subplot(222)
plt.scatter(X_train,y_train,label="train")
plt.scatter(X_test,predict["Lasso"][0],label="Ridge test")
plt.legend()

plt.subplot(223)
plt.scatter(X_train,y_train,label="train")
plt.scatter(X_test,predict["PolynomialRegression"][0],label="Ridge test")
plt.legend()

plt.show()










