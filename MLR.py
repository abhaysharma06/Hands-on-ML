#MLR 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv("Salary_Classification.csv")
features=dataset.iloc[:,:-1].values
labels=dataset.iloc[:,-1].values

dataset.isnull().any(axis=0)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder='passthrough')
features=np.array(ct.fit_transform(features),dtype=np.str)
features=features[:,1:]

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(features_train,labels_train)

labels_pred=regressor.predict(np.array(features_test,dtype='float64'))

x=[0,0,1500,1,2]
x=np.array(x)
x=x.reshape(1,5)
regressor.predict(x)