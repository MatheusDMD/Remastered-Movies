import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence
from tensorflow import keras
from time import time

def create_model(input_shape):
    model = keras.Sequential([


        keras.layers.Conv2D(64, (3,3), input_shape=input_shape, activation=tf.nn.relu, padding='same'),

        keras.layers.Conv2D(64,(3, 3), activation = tf.nn.relu, padding='same'),

        # keras.layers.Conv2D(128,(3, 3), activation = tf.nn.relu, padding='same'),
        # keras.layers.Conv2D(128,(3, 3), activation = tf.nn.relu, padding='same'),

        # keras.layers.Conv2D(256,(3, 3), activation = tf.nn.relu, padding='same'),
        # keras.layers.Conv2D(256,(3, 3), activation = tf.nn.relu, padding='same'),

        # keras.layers.Conv2D(512,(3, 3), activation = tf.nn.relu, padding='same'),        
        # keras.layers.Conv2D(256,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(128,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(64,(3, 3), activation = tf.nn.relu, padding='same'),

        keras.layers.Conv2D(32,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(3, (3, 3), activation = tf.nn.relu, padding='same'),

    ])
    
    model.compile(optimizer='adam', 
              loss='mse',
              metrics=['accuracy'])
    
    return model


img = cv2.imread('./dataset/y/2.jpg')

model=create_model(img.shape)
model.load_weights('./model/')

a = model.predict([[img]])
print(a.shape)
cv2.imwrite('result.jpg', a[0])