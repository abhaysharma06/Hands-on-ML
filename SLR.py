import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df=pd.read_csv("Salary_Data.csv")
features = df.iloc[:,:-1].values
labels=df.iloc[:,-1].values

df.isnull().sum()

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(features_train,labels_train)

labels_pred=regressor.predict(features_test)
regressor.predict([[6.9]])
regressor.predict(np.array(6.9).reshape(1,1))


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,labels_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pred)




