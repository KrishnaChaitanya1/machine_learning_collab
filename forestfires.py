[# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 08:22:42 2017

@author: Krishna
"""

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
# -------------- load dataset
url = r'forestfires.csv'
#names = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 
#        'wind', 'rain', 'area']
data = pd.read_csv(url)
#print(data.head(30)) # X --> x-axis spatial coordinate within the Montesinho park map: 1 to 9
# Y --> y-axis spatial coordinate within the Montesinho park map: 2 to 9
# ISI --> Initial Spread Index
# DC --> Drought Code
# DMC --> Duff Moisture Code
# FFMC --> Fine Fuel Moisture Code
# RH --> Relative Humidity
#print(data.keys())
#print(data.values)
df = pd.DataFrame(data)
df.drop(['month', 'day'], axis=1, inplace = True)
print(df.head())

# --------------- Splitting
array = df.values
X_data = array[:,0:10]
Y_data = array[:,10]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X_data, Y_data, test_size=validation_size, random_state=seed)
#scoring = 'accuracy'

# Model Selection
models = []
models.append(('LR', LinearRegression()))
models.append(('DTR', DecisionTreeRegressor()))
models.append(('SVR', SVR()))
models.append(('XBG', XGBRegressor()))
models.append(('RFR', RandomForestRegressor()))
models.append(('ETR', ExtraTreesRegressor()))
models.append(('GBR', GradientBoostingRegressor()))
# K-Fold

results = []
names = []
for name, model in models:
    kf = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train)
    results.append(cv_results)
    names.append(name)
    disp_txt = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
    print(disp_txt)
    
# Preditction
svr = SVR()
svr.fit(X_train, Y_train)
predict = svr.predict(X_validation)
print(mean_squared_error(Y_validation, predict))
