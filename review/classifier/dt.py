import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay

)
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

d_model = DecisionTreeClassifier(random_state=42)

params = {
    "max_depth": [3, 4, 5, 6, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}

dt_search = GridSearchCV(
    estimator=d_model,
    param_grid=params,
    cv=cv,
    scoring="f1"
)

dt_search.fit(X_train,y_train)

best_dt = dt_search.best_estimator_
dt_pred = best_dt.predict(X_test)

rf = RandomForestClassifier(random_state=42)
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 8, None],
    "max_features": ["sqrt", "log2", None]
}

rf_search = GridSearchCV(
    rf,
    param_grid=rf_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
rf_pred = best_rf.predict(X_test)

fig,axes = plt.subplots(2,2,figsize=(8,8))

ConfusionMatrixDisplay.from_predictions(y_test,dt_pred,ax=axes[0, 0])
axes[0, 0].set_title("DT Confusion Matrix")
RocCurveDisplay.from_predictions(
    y_true=y_test,
    y_pred=dt_pred,
    ax=axes[0, 1]
)
axes[0, 1].set_title("DT ROC")

ConfusionMatrixDisplay.from_predictions(y_test,rf_pred,ax=axes[1, 0])
axes[1, 0].set_title("RF Confusion Matrix")
RocCurveDisplay.from_predictions(
    y_true=y_test,
    y_pred=rf_pred,
    ax=axes[1, 1]
)
axes[1, 1].set_title("RF ROC")

plt.tight_layout()
plt.show()
