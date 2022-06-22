import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('affairs.csv')

features=df.iloc[:,:-1].values
labels=df.iloc[:,-1].values

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
features[:,0]=le.fit_transform(features[:,6])
features[:,0]=le.fit_transform(features[:,7])

one=OneHotEncoder(handle_unknown='error')
one.fit_transform(features[:,6:8].reshape(-1,1)).toarray()

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train,labels_train)

labels_pred =classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,labels_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pred)