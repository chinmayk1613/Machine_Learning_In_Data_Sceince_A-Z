# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:59:13 2020

@author: Chinmay Kashikar
"""


#Random Forest Regressor

#import library
import numpy as np   #cotain maths
import matplotlib.pyplot as plt #help to plot nice chart.To plot something
import pandas as pd #to import dataset and to manage data set

#import dataset
dataset=pd.read_csv('Position_Salaries.csv') #load data set
X=dataset.iloc[:,1:2].values    #independnt variables
y=dataset.iloc[:, 2].values     #dependent data

#Splitting dataset into train and test data 
#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""



#Fitting The Random Forest  Model to the Dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

#How to decide on n_estimators???

#Predicting a new result with Polynomial Regression
y_pred=regressor.predict([[6.5]])



#Visualising the Random Forest Regression Results(For higher resolution and smooth curve)
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid), 1)) 
plt.scatter(X,y,color='red')#real observation points
plt.plot(X_grid, regressor.predict((X_grid)),color='blue')#predicted salary by model
plt.title('Truth Or Bluff (Random Forest Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()