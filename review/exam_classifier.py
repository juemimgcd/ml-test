from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv(r"D:\machineL\ml-test\review\test_data\wdbc.csv")
X = pd.concat([data.iloc[:,0:1],data.iloc[:,2:]],axis=1)
y = data.iloc[:,1]
y = y.map({'M': 1, 'B': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)



models = {
    "dt":DecisionTreeClassifier(),
    "rf":RandomForestClassifier(),
    "bayes":GaussianNB(),
    "logic":LogisticRegression(max_iter=1000),
}

model_scores = {}

for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    p_score = precision_score(y_test,y_pred)
    r_score = recall_score(y_test,y_pred)
    f_score = f1_score(y_test,y_pred)
    model_scores[name] = [p_score,r_score,f_score]


for name,scores in model_scores.items():
    print(f"{name}:precision:{scores[0]} recall:{scores[1]} f1:{scores[2]}")




















