# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:16:37 2020
@author: Chinmay KAshikar
"""

#Simple Linear Regression   y=b0+b1*x1
#import library
import numpy as np   #cotain maths
import matplotlib.pyplot as plt #help to plot nice chart.To plot something
import pandas as pd #to import dataset and to manage data set

#import dataset
dataset=pd.read_csv('Salary_Data.csv') #load data set
X=dataset.iloc[:,:-1].values    #independnt variables
y=dataset.iloc[:, 1].values     #dependent data

#Splitting dataset into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train)"""
#scalling the dummy coulmn depends interpreation of model and it is depends

#Fitting Simple linear Regression to the train set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)  #ft is method name which fit the regreesor to data

#Predicting the Test set results
y_pred=regressor.predict(X_test)

#Visualising the Training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary VS Experience (Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()

#Visualising the Test set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary VS Experience (Test Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()


