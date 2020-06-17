# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:16:37 2020

@author: Chinmay Kashikar
"""
#Multiple Linear Regression

#import library
import numpy as np   #cotain maths
import matplotlib.pyplot as plt #help to plot nice chart.To plot something
import pandas as pd #to import dataset and to manage data set

#import dataset
dataset=pd.read_csv('50_Startups.csv') #load data set
X=dataset.iloc[:,:-1].values    #independnt variables
y=dataset.iloc[:, 4].values     #dependent data


#Encoding categorical data#Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
ct=ColumnTransformer([('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

# Avoiding Dummy variable Trap
X=X[:,1:]

#Splitting dataset into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""
#scalling the dummy coulmn depends interpreation of model and it is depends
#Always take no of dummy varibles-1 to avoid dummy variable trap 
#degrees of freedom...VC dimention...break point-1

#Fit Multiple Linear regression to train set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test set result
y_pred=regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.api as sm
#add coulum x0=1
#X_opt=np.array(X[:,[0,1,2,3,4,5]],dtype=float)
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#X_opt=X[:,[0,1,2,3,4,5]]
X_opt=np.array(X[:,[0,1,2,3,4,5]],dtype=float)
regreesor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regreesor_OLS.summary()
#here x2 varible have high p value greater than SL..
#So we have to remove x2 from model.

X_opt=np.array(X[:,[0,1,3,4,5]],dtype=float)
regreesor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regreesor_OLS.summary()

#remove x1
X_opt=np.array(X[:,[0,3,4,5]],dtype=float)
regreesor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regreesor_OLS.summary()

#remove x4
X_opt=np.array(X[:,[0,3,5]],dtype=float)
regreesor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regreesor_OLS.summary()

#remove x5
X_opt=np.array(X[:,[0,3]],dtype=float)
regreesor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regreesor_OLS.summary()

#Backward Elimination with p-values only
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

#Backward Elimination with p-values and Adjusted R Squared
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


