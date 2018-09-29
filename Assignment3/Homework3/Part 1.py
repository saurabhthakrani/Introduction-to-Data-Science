#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 16:43:14 2018

@author: saurabhthakrani
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('qb.train.csv')
x_train = dataset.iloc[:, [1, 6, 7, 9]].values
y = dataset.iloc[:, 8].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

x_train[:, 1] = labelencoder_X.fit_transform(x_train[:, 1])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x_train[:, 2] = labelencoder_X.fit_transform(x_train[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1, 2], sparse = False)
x_train = onehotencoder.fit_transform(x_train)

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y[:,])

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred) 

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred)

from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, Y_train)
 

y_pred1 = tree_model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred1)

from sklearn.svm import SVC
svm_class = SVC(kernel = 'rbf', random_state = 0)

svm_class.fit(X_train, Y_train)
y_pred2 = svm_class.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred2)
