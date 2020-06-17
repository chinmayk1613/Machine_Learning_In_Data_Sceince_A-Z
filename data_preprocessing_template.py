# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:04:43 2020

@author: Chinmay Kashikar
"""
#import library

import numpy as np   #cotain maths
import matplotlib.pyplot as plt #help to plot nice chart.To plot something
import pandas as pd #to import dataset and to manage data set



#import dataset

dataset=pd.read_csv('Data.csv') #load data set
X=dataset.iloc[:,:-1].values    #independnt variables
y=dataset.iloc[:, 3].values     #dependent data

#taking care of missing data
from sklearn.impute import SimpleImputer  #SimpleImputer class allow to take care of missing data
imputer=SimpleImputer(missing_values=np.nan,strategy='mean') # creating the object of class whihc take care of missing values providing some parameteres
imputer=imputer.fit(X[:, 1:3]) # fit the data to specific column
X[:,1:3]=imputer.transform(X[:,1:3]) #transform replcae the missing the data with mean of the column

#if we have caterical data we have to encode it from text to
#some number for machine learning models

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
#here problem is that machine learning algo thinks that 0<2 meaning
# France is less than spain but this is not the case at all
#hence we use dummy column buidling three column
#meanig put 1 if that France is there for ex. and put 0 if not.
ct=ColumnTransformer([('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

labelencoder_Y=LabelEncoder()
y=labelencoder_X.fit_transform(y)

#Splitting dataset into train and test data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#we should make feature scale to not let dominate one feature over the 
#other

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
#scalling the dummy coulmn depends interpreation of model and it is depends

