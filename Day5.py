
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df=pd.read_csv("Social_Network_Ads.csv")
features = df.iloc[:,2:4].values
labels=df.iloc[:,-1].values

#Spilitting dataset
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.5,random_state=0)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.fit_transform(features_test)

#KNN 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(features_train,labels_train)

labels_pred =classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pred)






