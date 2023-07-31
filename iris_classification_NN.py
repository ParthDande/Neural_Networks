# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 17:40:02 2023

@author: Parth
"""
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


#first we will load the data and perfrom preprocessing operators

iris = load_iris()
X,y = load_iris(return_X_y=True)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)

encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

#now we define the neural network

model = tf.keras.Sequential([tf.keras.layers.Dense(10,activation='relu',input_shape=(4,)),
                             tf.keras.layers.Dense(3,activation='softmax')])


#compile the model

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['accuracy'])


#train the model
model.fit(X_train,y_train_encoded,epochs = 100,batch_size =8,validation_split=0.1)


#evaluate the model

loss,accuracy = model.evaluate(X_test,y_test_encoded)
print(f"test accuracy:{accuracy}")