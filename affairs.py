import pandas as pd
import numpy as np
dataset = pd.read_csv("affairs.csv")

features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,-1].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[6])],remainder='passthrough')
features=np.array(ct.fit_transform(features),dtype=np.str)
features=features[:,1:]

from sklearn.compose import ColumnTransformer
ct1=ColumnTransformer([('encoder',OneHotEncoder(),[11])],remainder='passthrough')
features=np.array(ct1.fit_transform(features),dtype=np.str)
features=features[:,1:]

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .25, random_state = 0) 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)



from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(features_train, labels_train)

probability = classifier.predict_proba(features_test)

labels_pred = classifier.predict(features_test)

pd.DataFrame(labels_pred, labels_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)
#Accuracy Score
from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pred)



import numpy as np
"""arr = [0,1,0,0,0,0,0,0,1,0,3,25,3,1,4,16]
arr = list(np.array(arr).reshape(1,-1)) """  

x = [3,25,3,1,4,16,4,2]
x = np.array(x).reshape(1,8)
y=np.array(ct.transform(x),dtype=np.str)
y = y[:,1:]

z = np.array(ct1.transform(y),dtype=np.str)
z = z[:,1:]
z=np.array(z,dtype=np.float)

result = classifier.predict_proba(z)
result1 = classifier.predict(z)
arr = list(np.array(z).reshape(1,-1)) 



xm =[3,32,9,3,3,17,2,5]
xm = np.array(xm).reshape(1,8)
ym=np.array(ct.transform(xm),dtype=np.str)
ym = ym[:,1:]

z = np.array(ct1.transform(ym),dtype=np.str)
z = z[:,1:]
z=np.array(z,dtype=np.float)

result = classifier.predict_proba(z)
result1 = classifier.predict(z)
arr = list(np.array(z).reshape(1,-1)) 













