# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 17:16:37 2020

@author: cskck
"""


#import library
import numpy as np   #cotain maths
import matplotlib.pyplot as plt #help to plot nice chart.To plot something
import pandas as pd #to import dataset and to manage data set

#import dataset
dataset=pd.read_csv('Data.csv') #load data set
X=dataset.iloc[:,:-1].values    #independnt variables
y=dataset.iloc[:, 3].values     #dependent data

#Splitting dataset into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""
#scalling the dummy coulmn depends interpreation of model and it is depends
