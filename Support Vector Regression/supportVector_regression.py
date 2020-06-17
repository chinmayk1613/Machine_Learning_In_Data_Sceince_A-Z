# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 13:56:44 2020

@author: Chinmay Kashikar
"""


#Support Vector Regression

#import library
import numpy as np   #cotain maths
import matplotlib.pyplot as plt #help to plot nice chart.To plot something
import pandas as pd #to import dataset and to manage data set

#import dataset
dataset=pd.read_csv('Position_Salaries.csv') #load data set
X=dataset.iloc[:,1:2].values    #independnt variables
y=dataset.iloc[:, 2:].values     #dependent data

#Splitting dataset into train and test data 
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)



#Fitting The SVR to the Dataset
#Create your regressor
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)

#Predicting a new result with Polynomial Regression
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.fit_transform(np.array([[6.5]]))))


#Visualising the SVR Results
plt.scatter(X,y,color='red')#real observation points
plt.plot(X, regressor.predict((X)),color='blue')#predicted salary by model
plt.title('Truth Or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Visualising the SVR Results(For higher resolution and smooth curve)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid), 1)) 
plt.scatter(X,y,color='red')#real observation points
plt.plot(X_grid, regressor.predict((X_grid)),color='blue')#predicted salary by model
plt.title('Truth Or Bluff (SVR Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
