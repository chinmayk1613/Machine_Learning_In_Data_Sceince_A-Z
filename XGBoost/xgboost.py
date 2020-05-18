# -*- coding: utf-8 -*-
"""
Created on Mon May 18 22:58:51 2020

@author: Chinmay Kashikar
"""
#import library
import numpy as np   
import matplotlib.pyplot as plt 
import pandas as pd 
import xgboost as xgb


#import dataset
dataset=pd.read_csv('Churn_Modelling.csv') 
X=dataset.iloc[:,3:13].values    
y=dataset.iloc[:, 13].values     


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X=LabelEncoder()
X[:,1]=labelencoder_X.fit_transform(X[:,1])

labelencoder_X2=LabelEncoder()
X[:,2]=labelencoder_X2.fit_transform(X[:,2])


#here problem is that machine learning algo thinks that 0<2 meaning
# France is less than spain but this is not the case at all
#hence we use dummy column buidling three column
#meanig put 1 if that France is there for ex. and put 0 if not.
ct=ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

X=X[:,1:]

#Splitting dataset into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#Fitting XGboots to training set
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train, y_train)

#Predicting the test set result
y_pred=classifier.predict(X_test)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Applying k-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train,y=y_train,cv=10)
mean=accuracies.mean()
std=accuracies.std()




