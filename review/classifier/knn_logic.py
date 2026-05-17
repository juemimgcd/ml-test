import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

k_model = Pipeline([
    ("scaler",StandardScaler()),
    ("model",KNeighborsClassifier())
])

k_params = {
    "model__n_neighbors":[3,5,7,9,11]
}

k_search = GridSearchCV(
    estimator=k_model,
    param_grid=k_params,
    scoring="f1",
    cv=cv,

)

k_search.fit(X_train,y_train)
best_knn = k_search.best_estimator_
knn_pred = best_knn.predict(X_test)

# print(best_knn.score(X_test,y_test))


l_model = Pipeline([
    ("scaler",StandardScaler()),
    ("model",LogisticRegression())
])

l_params = {
    "model__C": np.logspace(-3, 2, 20).tolist(),
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear"]
}

l_search = RandomizedSearchCV(
    estimator=l_model,
    param_distributions=l_params,
    cv=cv,
    scoring="f1",
    random_state=42,
    n_iter=10
)
l_search.fit(X_train, y_train)
best_l = l_search.best_estimator_
l_pred = best_l.predict(X_test)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
# ConfusionMatrixDisplay.from_predictions(
#     y_test, knn_pred, ax=axes[0], cmap="Blues", colorbar=False
# )
# axes[0].set_title("KNN Confusion Matrix")
#
# ConfusionMatrixDisplay.from_predictions(
#     y_test, l_pred, ax=axes[1], cmap="Oranges", colorbar=False
# )
# axes[1].set_title("Logistic Regression Confusion Matrix")
#
# plt.tight_layout()
# plt.show()





RocCurveDisplay.from_estimator(
    best_knn, X_test, y_test, name="Best KNN", color="tab:blue"
)
RocCurveDisplay.from_estimator(
    best_l, X_test, y_test, name="Best Logistic", color="tab:orange"
)

# 随机分类器基线
plt.plot([0, 1], [0, 1], "k--", label="Chance")

plt.title("ROC Curve on Test Set")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


