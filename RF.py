#Classification 
#Random Forest


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

#Importing the dataset
df=pd.read_csv("Social_Network_Ads.csv")
features=df.iloc[:,2:4].values
labels=df.iloc[:,-1].values

# Splitting the dataset 
from sklearn.model_selection import train_test_split
features_train , features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.25,random_state=40)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)   

features_test=sc.fit_transform(features_test)

#Fitting RF to the Training Set
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(features_train,labels_train)

#Predicting the test set 
labels_pred=classifier.predict(features_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test, labels_pred)


# Accuracy Score 
from sklearn.metrics import accuracy_score
ac=accuracy_score(labels_test,labels_pred)

from matplotlib.colors import ListedColormap
x_set,y_set=features_train,labels_train
x1,x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max() +1,step=0.01),
                    np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max() +1,step=0.01))

plt.contour(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title("RF (Training set)")    
plt.xlabel("age")
plt.ylabel("salary")
plt.legend()
plt.show()


from matplotlib.colors import ListedColormap
x_set,y_set=features_test,labels_test
x1,x2 = np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max() +1,step=0.01),
                    np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max() +1,step=0.01))

plt.contour(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
            alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title("RF (Training set)")    
plt.xlabel("age")
plt.ylabel("salary")
plt.legend()
plt.show()





