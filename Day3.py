'''
df = pd.read_csv("training_titanic.csv")
df.isnull().sum()
df = df.dropna()  # removing all rows containing nan values
df.dropna(how='all') 
df=df.dropna(thresh=11)  # drop rows if it have max  non-nan values.

df=df.dropna(subset=['Cabin'])  #drop nan in specific coloumn


df=df.fillna(0)

'''

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df= pd.read_csv('Data.csv')
features=df.iloc[:,0:3].values
labels=df.iloc[:,-1].values

df.isnull().sum()


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(features[:,1:3])
features[:,1:3]=imputer.transform(features[:,1:3])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
features[:,0]=le.fit_transform(features[:,0])

one=OneHotEncoder(handle_unknown='error')
one.fit_transform(features[:,0].reshape(-1,1)).toarray()


from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder='passthrough')
features=np.array(ct.fit_transform(features),dtype=np.str)

le=LabelEncoder()
labels=le.fit_transform(labels)


from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test =train_test_split(features,labels,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.fit_transform(features_test)












