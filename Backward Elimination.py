#Backward Elimination 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

dataset = pd.read_csv("Salary_Classification.csv")
features=dataset.iloc[:,:-1].values
labels=dataset.iloc[:,-1].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder='passthrough')
features=np.array(ct.fit_transform(features),dtype=np.str)
features=features[:,1:]

import statsmodels.regression.linear_model as sm

'''
features =np.append(arr=np.ones((30,1)).astype(int),
                    values=features,axis=1)

features_opt=features[:,[0,1,2,3,4,5]]
labels=labels.astype(float)
features_opt=features_opt.astype(float)
ols =sm.OLS(endog=labels,exog=features_opt).fit()
ols.summary()

features_opt=features[:,[0,1,3,4,5]]
labels=labels.astype(float)
features_opt=features_opt.astype(float)
ols =sm.OLS(endog=labels,exog=features_opt).fit()
ols.summary()

features_opt=features[:,[0,1,3,5]]
labels=labels.astype(float)
features_opt=features_opt.astype(float)
ols =sm.OLS(endog=labels,exog=features_opt).fit()
ols.summary()

features_opt=features[:,[0,3,5]]
labels=labels.astype(float)
features_opt=features_opt.astype(float)
ols =sm.OLS(endog=labels,exog=features_opt).fit()
ols.summary()


features_opt=features[:,[0,5]]
labels=labels.astype(float)
features_opt=features_opt.astype(float)
ols =sm.OLS(endog=labels,exog=features_opt).fit()
ols.summary()
'''

features_obj = features[:,[0,1,2,3,4,]]
features_obj =np.append(arr=np.ones((30,1)).astype(int),
                    values=features,axis=1)

while True:
    features_obj=features_obj.astype(float)
    labels=labels.astype(float)
    regressor_ols = sm.OLS(endog=labels,exog=features_obj).fit()
    p_values=regressor_ols.pvalues
    if p_values.max() > 0.05:
        features_obj = np.delete(features_obj,p_values.argmax(),1)
    else:
        break






















from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)


