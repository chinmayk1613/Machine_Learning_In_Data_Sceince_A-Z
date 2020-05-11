# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:01:30 2020

@author: Chinmay Kashikar
"""


#Artificial Neural Network ANN

# Theano library---> runs on GPU to compute calculation in more faster way.
# Tenserflow ---> used for deep learning and deep neural networks.
# Keras ---> Wraps above two. can create deep neural network wiht very few lines of code.

#Part 1----> Data Preprocessing

#import library
import numpy as np   
import matplotlib.pyplot as plt 
import pandas as pd 

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


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
#scalling the dummy coulmn depends interpreation of model and it is depends


#Part 2 - Implementation of ANN
#First thing first... Importing Libraries
import keras
from keras.models import Sequential #To Initialize the ANN
from keras.layers import Dense # to create ANN layers

#Intialising the ANN
classifier= Sequential()

# Adding the first layer and first hidden layer
classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation= 'relu', input_dim=11)) #Configuration of hidden layers

# Adding the second hidden layer layer
classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation= 'relu')) #Configuration of second hidden layers

# Adding the Output Layer
classifier.add(Dense(units=1, kernel_initializer = 'uniform', activation= 'sigmoid')) #Configuration of output layers
#softmax used as activation function at output layer if we have more than 2 categories

# Compiling the ANN
classifier.compile(optimizer= 'adam', loss='binary_crossentropy', metrics= ['accuracy'])
#loss = categorical_crossentropy ifwe have more than 2 dependent variable/output

# Fitting the ANN to training set
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

# Part ----> 3 Make the prediction on test data
#Predicting the test set result
y_pred=classifier.predict(X_test)
y_pred=(y_pred > 0.5) #to convert preobabilty into category to feed this to confusion matrix

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)