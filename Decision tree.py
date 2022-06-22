#Classification 
#dEcision tree


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

#Importing the dataset
df=pd.read_csv("Social_Network_Ads.csv")
features=df.iloc[:,:-1].values
labels=df.iloc[:,-1].values

# Splitting the dataset 
from sklearn.model_selection import train_test_split
features_train , features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.25,random_state=40)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.fit_transform(features_test)

#Fitting KNN to the Training Set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(features_train,labels_train)

#Predicting the test set 
labels_pred=classifier.predict(features_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test, labels_pred)


# Accuracy Score 
from sklearn.metrics import accuracy_score
ac=accuracy_score(labels_test,labels_pred)


