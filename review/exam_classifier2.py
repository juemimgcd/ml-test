from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns



encoder = LabelEncoder()
data = pd.read_csv(r"D:\machineL\ml-test\review\test_data\titanic.csv")
data["Embarked"] = data["Embarked"].dropna()
# 选取特征并处理缺失值
# 选取数值特征和类别特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = data[features].copy()
y = data['Survived']

# 填充缺失值
X['Age'] = X['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())


# 类别特征编码
X['Sex'] = LabelEncoder().fit_transform(X['Sex'])
X['Embarked'] = LabelEncoder().fit_transform(X['Embarked'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


models = {
    "rf":RandomForestClassifier(),
    "xgb":XGBClassifier(),
    "bagging":BaggingClassifier(),
    "dt":DecisionTreeClassifier()
}

precision_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    print(f"{name}:{matrix}")
    precision_scores[name] = precision


plt.figure(figsize=(12, 6))
# 提取模型名称和对应的 precision 值
model_names = list(precision_scores.keys())
precision_values = list(precision_scores.values())


bars = plt.bar(model_names, precision_values, align='center')
for bar,score in zip(bars, precision_values):
    plt.text(bar.get_x(),bar.get_height() + 0.01,f"{score:.2f}", ha='center', va='bottom')
plt.show()

