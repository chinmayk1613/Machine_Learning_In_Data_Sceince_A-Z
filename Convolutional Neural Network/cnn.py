# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:55:49 2020

@author: Chinmay Kashikar
"""

#Convolutional Neural Network

#Part 1- Building the CNN

from keras.models import Sequential # to ini NN
from keras.layers import Convolution2D # for the convolution layer step
from keras.layers import MaxPooling2D # for the pooling layer step
from keras.layers import Flatten # to create large feature vector
from keras.layers import Dense # to create fully connected network

# Initialising the CNN
classifier= Sequential()

#Step 1 - Convolution Layer
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation = 'relu'))
#nb_filter= number of feature map=32
#no of rows=3 and no of columns=3
#input shape=force all input images to same format...
#256 256 3 tenserflow backend format

#Step 2- Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Step 3- Flattening
classifier.add(Flatten())

#Step 4- Full Connection of classic ANN
classifier.add(Dense(units= 128, activation='relu')) #Hidden layer
classifier.add(Dense(units= 1, activation='sigmoid')) #Output layer

# Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])


#Part 2 - Fitting the CNN to the Image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

tarining_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        tarining_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        nb_val_samples=2000)
