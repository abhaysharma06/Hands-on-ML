#Classification
#DAy9

import pandas as pd
from sklearn.datasets import load_iris
dataset= load_iris()
features=dataset['data']
labels=dataset['target']

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.30,random_state=0)

from sklearn.svm import SVC
classifier =SVC(kernel='rbf',random_state=0)
classifier.fit(features_train,labels_train)

labels_pred =classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pred)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(features_train,labels_train)
lables_pred=gnb.predict(features_test)

cm_gnb = confusion_matrix(labels_test,labels_pred)
ac_gnb = accuracy_score(labels_test,labels_pred)

