# Day 11

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

dataset =pd.read_csv("Claims_Paid.csv")
features=dataset.iloc[:,0:1].values
labels =dataset.iloc[:,1].values

#Visualize
plt.scatter(features,labels)
plt.show()

#fitting model with simple linear regression
from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(features,labels) 

lin_reg1.predict([[1981]])

plt.scatter(features,labels,color='red')
plt.plot(features,lin_reg1.predict(features),color='blue')
plt.title('Linear Regression')
plt.xlabel('Year')
plt.ylabel("Cost")
plt.show()

# Fitting our model with polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_object =PolynomialFeatures(degree = 4)
features_poly = poly_object.fit_transform(features)

lin_reg2 = LinearRegression()
lin_reg2.fit(features_poly,labels)

lin_reg2.predict(poly_object.transform([[1981]]))

#Visaulize the polynomial set 
#features_grid=np.arange(min(features),max(features),0.01)
#features_grid=features_grid.reshape(len(features_grid),1)
plt.scatter(features,labels,color='red')
plt.plot(features,lin_reg2.predict(poly_object.fit_transform(features)),color='blue')
plt.title('PLR')
plt.xlabel('Year')
plt.ylabel("Cost")
plt.show()

































