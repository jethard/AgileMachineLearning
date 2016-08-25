import pandas as pd
from pandas import Series,DataFrame
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


data = pd.read_csv("data.csv")
train, test = train_test_split(data, test_size = 0.2)
X_train = train.drop("Survived",axis=1)
Y_train = train["Survived"]
X_test = test.drop("Survived", axis = 1)
Y_test = test['Survived']
random_forest = RandomForestClassifier(n_estimators = 1000)
random_forest.fit(X_train, Y_train)
random_forest.score(X_test, Y_test)

def wrapper_for_titanic_Random_Forest_code(train_x, train_y, test_x, test_y):
    score = None

    random_forest = RandomForestClassifier(n_estimators = 1000)
    random_forest.fit(train_x, train_y)
    score = random_forest.score(test_x, test_y)

    return score
