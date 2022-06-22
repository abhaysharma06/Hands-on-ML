import pandas as pd 
import matplotlib.pyplot as plt

dataset =pd.read_csv("Position_Salaries.csv")
features = dataset.iloc[:,1:2].values
labels=dataset.iloc[:,2].values

plt.scatter(features,labels)

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
sc1 = StandardScaler()
features=sc.fit_transform(features)
labels=sc1.fit_transform(labels.reshape(-1,1))



from sklearn.svm import SVR 
regressor = SVR(kernel='rbf')
regressor.fit(features,labels)

sc1.inverse_transform(regressor.predict(sc.transform([[6.5]])))
plt.scatter(features,labels,color='red')
plt.plot(features,regressor.predict(features),color='blue')
plt.title('SVR')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


