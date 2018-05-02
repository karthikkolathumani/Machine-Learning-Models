# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:12:29 2018

@author: karth
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#Initializing the CNN
cls = Sequential()


cls.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation = 'relu'))

cls.add(MaxPooling2D(pool_size = (2,2)))

cls.add(Convolution2D(32, 3, 3, activation = 'relu'))

cls.add(MaxPooling2D(pool_size = (2,2)))

# Flattening 
cls.add(Flatten())

# Fully COnnected Layer
cls.add(Dense(output_dim = 128, activation = 'relu'))
cls.add(Dense(output_dim = 1, activation = 'sigmoid'))

##
cls.compile(optimizer = 'adam' , loss= 'binary_crossentropy', metrics = ['accuracy'])


#Fitting
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/Users/karth/Documents/ML/Project/Dataset/Cropped_train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:/Users/karth/Documents/ML/Project/Dataset/Test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

cls.fit_generator(
        training_set,
        steps_per_epoch=6500,
        epochs=25,
        validation_data=test_set,
        validation_steps=1500)

from keras.models import load_model

cls.save('first_model.h5') 

cls.save_weights('first_model_weights.h5')

