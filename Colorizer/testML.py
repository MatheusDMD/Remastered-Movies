import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence
from tensorflow import keras
from time import time

def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3,3), input_shape=input_shape, activation=tf.nn.relu, padding='same', strides=2),
        keras.layers.Conv2D(128,(3, 3), activation = tf.nn.relu, padding='same',strides=2),
        keras.layers.Conv2D(256,(3, 3), activation = tf.nn.relu, padding='same',strides=2),
        keras.layers.UpSampling2D(size=2),
        keras.layers.Conv2D(128,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.UpSampling2D(size=2),
        keras.layers.Conv2D(64,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.UpSampling2D(size=2),
        keras.layers.Conv2D(32,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(3, (3, 3), activation = tf.nn.relu, padding='same'),
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    return model


img = cv2.imread('./dataset/y_small/49.jpg')

checkpoint_path = "model/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)

model=create_model(img.shape)
model.load_weights("model/cp-1100.ckpt")

a = model.predict([[img]])
print(a.shape)
cv2.imwrite('result.jpg', a[0])