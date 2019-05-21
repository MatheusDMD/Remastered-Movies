import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence
from tensorflow import keras
from time import time
import numpy as np

def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Conv2D(64,(3, 3),input_shape=input_shape, activation = tf.nn.relu, padding='same', strides=2),
        keras.layers.Conv2D(128,(3, 3), activation = tf.nn.relu, padding='same',strides=2),
        keras.layers.Conv2D(256,(3, 3), activation = tf.nn.relu, padding='same',strides=2),
        keras.layers.Conv2D(512,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(512,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(512,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(512,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.UpSampling2D(size=2),
        keras.layers.Conv2D(256,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(128,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.UpSampling2D(size=2),
        keras.layers.Conv2D(64,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(2, (3, 3), activation = tf.nn.tanh, padding='same'),
        keras.layers.UpSampling2D(size=2),
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    return model



checkpoint_path = "model/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)


# image = cv2.imread(file_name_y)
image = cv2.imread('./dataset/y/16.jpg')
lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
l = lab[:,:,:1]
print(lab)
print(l.shape)
model=create_model(l.shape)
model.load_weights(latest)
# ab = lab[:,:,1:]

output = model.predict([[l]])
# print(a.shape)
cur = np.zeros((256, 256, 3))
cur[:,:,0] = l[:,:,0]
cur[:,:,1:] = output
cv2.imwrite("result.png", cur)