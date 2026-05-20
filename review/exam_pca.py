import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns



data = pd.read_csv(r"D:\machineL\ml-test\review\test_data\titanic.csv")
data["Embarked"] = data["Embarked"].dropna()
# 选取特征并处理缺失值
# 选取数值特征和类别特征
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X = data[features].copy()
y = data['Survived']

# 填充缺失值
X['Age'] = X['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())





# 2. 数据标准化（PCA降维前的关键步骤）
# PCA是基于方差的算法，不同量纲的特征（如年龄和票价）会影响降维结果，必须先标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(X.shape)
print(pca.n_components_)
print(pca.explained_variance_)











