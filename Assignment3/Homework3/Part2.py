#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:09:27 2018

@author: saurabhthakrani
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('qb.train.csv')

import seaborn as sns

import re
pattern = "^[0-9][0-9][0-9][0-9]"

for index, row in df.iterrows():
    temp = df['tournaments'].values[index]
    year = re.match(pattern, temp)
    df['tournament_year'].values[index] = year.group(0)
    
import re
for index,row in df.iterrows():
    length=df['text'].values[index]
    length=length.split()
    length=len(length)
    df['length'].values[index]=length

X=df.iloc[:,[1,6,7,9,10,11]].values

y=df.iloc[:,  8].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

label_encoder_1=LabelEncoder()
label_encoder_1.fit(X[:, 1])
X[:, 1]=label_encoder_1.transform(X[:, 1])

label_encoder_2=LabelEncoder()
label_encoder_2.fit(X[:, 2])
X[:, 2]=label_encoder_2.transform(X[:, 2])

label_encoder_corr=LabelEncoder()
label_encoder_corr.fit(y)
y=label_encoder_corr.transform(y)

one_hot_encoder=OneHotEncoder(categorical_features=[1,2],sparse=False)
one_hot_encoder.fit(X)
X=one_hot_encoder.transform(X)

X=np.delete(X,one_hot_encoder.feature_indices_[: -1],1)

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_test_show = X_test

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

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(Y_test, y_pred1) 


from sklearn.svm import SVC
svm_class = SVC(kernel = 'rbf', random_state = 0)

svm_class.fit(X_train, Y_train)
y_pred2 = svm_class.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred2)


from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(Y_test, y_pred2) 

print(confusion_matrix(Y_test, y_pred1))


test=pd.read_csv('qb.test.csv')


import re
pattern = "^[0-9][0-9][0-9][0-9]"

for index, row in test.iterrows():
    temp = test['tournaments'].values[index]
    year = re.match(pattern, temp)
    test['tournament_year'].values[index] = year.group(0)
    
import re
for index,row in test.iterrows():
    length=test['text'].values[index]
    length=length.split()
    length=len(length)
    test['length'].values[index]=length    
    
features=test.iloc[:,[1,5,6,7,8,9]].values

features[:, 1]=label_encoder_1.transform(features[:, 1])

features[:, 2]=label_encoder_2.transform(features[:, 2])


features=one_hot_encoder.transform(features)

features=np.delete(features,one_hot_encoder.feature_indices_[: -1],1)


from sklearn.tree import DecisionTreeClassifier
tree_model1 = DecisionTreeClassifier()
tree_model1.fit(X, y)


y_pred_final=tree_model1.predict(features)
y_pred_prob=tree_model1.predict_proba(features)[:,1]

final_df=pd.DataFrame(data=y_pred_final)
final_df['row']=test['row']
final_df.columns=['corr','row']
final_df.to_csv('predictions_final.csv',encoding='utf-8',index=False)
