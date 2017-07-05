# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 02:01:20 2017

@author: Krishna
"""

# --------------- Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# --------------- Load dataset
url= r'C:\Users\kchai\Dropbox\ML Root\The Journey of an Idiot in Machine Learning\Datasets\Titanic Dataset\train.csv'
data = pd.read_csv(url)
train = pd.get_dummies(data, columns=['Sex'])
df_train = pd.DataFrame(train)
#print(df_train.head())
df_train.dropna(inplace = True)
df_train.drop(['PassengerId', 'Name', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1, inplace = True)
#print(df_train)

# ----------------- Splitting
Y = df_train['Survived']
X = df_train.drop(['Survived'], axis=1)
#print(X.head())
x_train, x_cv, y_train, y_cv = train_test_split(X, Y, test_size=0.3)
#print(x_train.head())

# ------------------- Testing Data
url1= r'C:\Users\kchai\Dropbox\ML Root\The Journey of an Idiot in Machine Learning\Datasets\Titanic Dataset\test.csv'
data_test = pd.read_csv(url1)
#print(data_test.keys())
test = pd.get_dummies(data_test, columns=['Sex'])
df_test = pd.DataFrame(test)
#print(df_test.head())
df_test.dropna(inplace = True)
X_test = df_test.drop(['PassengerId', 'Name', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1, inplace = True)
#print(X_test.head())
# ------------------- Model Fitting --> CV test
lgr = LogisticRegression()
lgr.fit(x_train, y_train)
predict_lgr = lgr.predict(x_cv)
mse = np.mean((predict_lgr - y_cv)**2)
#print(mse)

# ------------------ Model Fitting --> Test
predict_lgr_test = lgr.predict(X_test)
mse_test = np.mean((predict_lgr_test - y_cv)**2)
print(mse_test)