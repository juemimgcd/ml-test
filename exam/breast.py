from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay,precision_score
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data["data"]
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"测试：{len(X_test), len(y_test)},训练：{len(X_train), len(y_train)}")

models = {
    "logic": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(),
    "xg": XGBClassifier()
}

fig, axes = plt.subplots(1, 3, figsize=(10, 8))
ax = axes.flatten()
i = 0

model_a_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_a_scores[name] = accuracy_score

    print(f"{name},score:{accuracy}")
    ConfusionMatrixDisplay.from_predictions(
        y_true=y_test,
        y_pred=y_pred,
        ax=ax[i],

    )
    ax[i].set_title(f"{name},confusion_matrix_graph")
    i += 1
    if i == 3:
        break
plt.show()