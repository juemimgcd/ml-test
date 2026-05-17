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
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier


data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

arr1 = []
arr2 = []


k_values = [2, 3, 4]
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for k in k_values:
    d_model = DecisionTreeClassifier(max_depth=k, random_state=42)
    # Cross-validate on the training data, not on predictions
    a_score = cross_val_score(d_model, X_train, y_train, cv=cv, scoring="recall").mean()
    p_score = cross_val_score(d_model, X_train, y_train, cv=cv, scoring="precision").mean()
    arr1.append(a_score)
    arr2.append(p_score)


plt.subplot(2,2,1)
plt.plot(k_values, arr1, marker="o", label="recall")
plt.xlabel("max_depth")
plt.ylabel("score")
plt.legend()

plt.subplot(2,2,2)
plt.plot(k_values, arr2, marker="o", label="precision")
plt.xlabel("max_depth")
plt.ylabel("score")
plt.legend()

plt.subplot(2,2,3)
plt.plot(arr1,arr2,label="precision and recall")
plt.xlabel("precision")
plt.ylabel("recall")


plt.tight_layout()
plt.show()
