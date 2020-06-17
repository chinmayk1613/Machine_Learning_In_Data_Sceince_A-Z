# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:02:46 2020

@author: Chinmay Kashikar

"""

#import library
import numpy as np   #cotain maths
import matplotlib.pyplot as plt #help to plot nice chart.To plot something
import pandas as pd #to import dataset and to manage data set

#import dataset
dataset=pd.read_csv('Social_Network_Ads.csv') #load data set
X=dataset.iloc[:,[2,3]].values    #independnt variables
y=dataset.iloc[:, 4].values     #dependent data

#Splitting dataset into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
#scalling the dummy coulmn depends interpreation of model and it is depends

#Fitting the KNN classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

#Predicting the test set result
y_pred=classifier.predict(X_test)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Visualising the Training set Results
from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],
    c=ListedColormap(('red','green'))(i),label=j)
plt.title('KNN Classifier (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()    


#Visualising the Test set Results
from matplotlib.colors import ListedColormap
X_set,y_set=X_test,y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1, stop=X_set[:,0].max()+1,step=0.01),
                  np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],
    c=ListedColormap(('red','green'))(i),label=j)
plt.title('KNN Classifier (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()  