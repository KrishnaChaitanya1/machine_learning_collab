# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:02:38 2017

@author: Krishna
"""
# Import the necessary packages
import numpy as np
# from the sklearn package import DecisionTreeClassifier subclass from tree class
from sklearn.tree import DecisionTreeClassifier
# load the data. 'r' signifies to just read the data.
f = open('decision_tree_data.txt', 'r')
x_train = [] # Create an empty list of X_train data
y_train = [] # Create an empty list of y_train data

# iterate over line in the data.
for line in f:
    line = np.asarray(line.split(), dtype=np.float32) # store the data in an array and split it.
    x_train.append(line[:-1]) # append x_train from Age till Marital Status. refer the data in the assignment
    y_train.append(line[-1]) # append y_train only the target variable, Buys

x_train = np.asmatrix(x_train) # represent x_train as a matrix
y_train = np.reshape(y_train, (len(y_train),1)) # reshape the output array to fit the classifier
clf = DecisionTreeClassifier(random_state=0) # create an object for the Classifier. random state is made 0to make sure the randomness is Zero
fit = clf.fit(x_train, y_train) # fit the classifier
predict = clf.predict(x_train) # predict on x_train. Since the test data is not given and the train data is too small to partition, we take the x_train as the test value
print(predict)