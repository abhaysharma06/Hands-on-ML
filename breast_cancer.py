# -*- coding: utf-8 -*-
"""
Created on Fri May 22 22:27:03 2020

@author: Rajat arya
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("breast_cancer.csv")
dataset.isnull().any(axis = 0)

#from scipy import stats
#mode_ = stats.mode(dataset['G'])
dataset['G']=dataset['G'].fillna(dataset['G'].mode()[0])
dataset.info()

#dataset['G'].value_counts()
#dataset['G'] = dataset['G'].fillna(1)

features = dataset.iloc[:,1:-1].values
labels = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .25, random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(features_train, labels_train)

labels_pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)

score = classifier.score(features_test,labels_test)
score1 = classifier.score(features_train,labels_train)

dataset["tumor"] = dataset["K"].map(lambda x: 'Benign' if x == 2 else 'Malignant' )

a = [6,2,5,3,2,7,9,2,4]
a = np.array(a).reshape(1,-1)
result = classifier.predict(a)
