import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

red_wine = pd.read_csv("winequality-red.csv",sep=";")

# red_wine as its (feature+labels) type
red_wine.dtypes

#describe the red_wine as "count", "mean", "min" ,"max" and etc.
red_wine.describe()

import seaborn as sns
sns.boxplot(red_wine["residual sugar"])     #boxplot is used to visualize the distribution of the values in ‘residual sugar’
                                            # actual maximum value is around 4 and more than this are outliner which we do eleminate
plt.subplots(figsize=(15, 10))
sns.heatmap(red_wine.corr(), annot = True, cmap = "coolwarm")   #correlation between attributes
                                    

from scipy import stats                            #Eliminating outliers by Z-score
z = np.abs(stats.zscore(red_wine))
red_wine = red_wine[(z < 3).all(axis=1)]
red_wine.shape                                        

# checking the class imbalance                                        
red_wine["quality"].value_counts()

# Define features X
X = red_wine.iloc[:,:-1].values                  # data  pre preprocesing
# Define labels y
y = red_wine.iloc[:,-1].values

from sklearn import preprocessing                        # Standardizing the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split      # Train and Test dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)
 
# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(multi_class="multinomial",solver ="newton-cg")  # Train and fit model
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)                  # Predict out-of-sample test set

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)          #accuracy_score = 58%

from sklearn.model_selection import train_test_split      # Train and Test dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

#  KNN Train the model and predict for k=35 accuracry is 56.01%
from sklearn.neighbors import KNeighborsClassifier                   
knn = KNeighborsClassifier(n_neighbors=35)        
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)



from sklearn.model_selection import train_test_split      # Train and Test dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

# Train and fit the Decision Tree Classification model  
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
# Evaluate the model with out-of-sample test set
y_pred = tree.predict(X_test)
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

# Accuracy Score 
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)   # accuracy_score =  0.6219931271477663



from sklearn.model_selection import train_test_split      # Train and Test dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0)

# Train and fit the Random Forest Classification model
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100,random_state = 0)
forest.fit(X_train, y_train)
# Test out-of-sample test set
y_pred = forest.predict(X_test)
#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

# Accuracy Score 
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)           # accuracy_score =  0.711340206185567
