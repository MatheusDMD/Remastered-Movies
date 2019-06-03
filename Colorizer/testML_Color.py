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
        keras.layers.Conv2D(64,(3, 3),input_shape=input_shape, padding='same', strides=2),
        keras.layers.Conv2D(128,(3, 3), activation = tf.nn.relu, padding='same',strides=2),
        keras.layers.Conv2D(256,(3, 3), activation = tf.nn.relu, padding='same', strides=2),
        keras.layers.UpSampling2D(size=2),
        keras.layers.Conv2D(256,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.UpSampling2D(size=2),
        keras.layers.Conv2D(128,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.UpSampling2D(size=2),
        keras.layers.Conv2D(64,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(2, (3, 3), activation = tf.nn.tanh, padding='same'),
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    return model



checkpoint_path = "model/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)

for i in ["0005", "0010"]:

    checkpoint_path = f"./model/cp-{i}.ckpt"
    
    image = cv2.imread('./dataset/z/16.jpg')
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l = lab[:,:,:1] / 255.0 - 1.0
    ab = lab[:,:,1:]
    print(ab)
    model=create_model(l.shape)
    model.load_weights(checkpoint_path)
    output = model.predict([[l]]) * 255 + 1

    cur = np.ones((256, 256, 3))
    cur[:,:,0] = lab[:,:,0]
    cur[:,:,1:] = output
    cur = cur.astype('uint8')
    print(output)
    print(lab[:,:,0])
    print(cur.shape)
    final = cv2.cvtColor(cur, cv2.COLOR_LAB2RGB)
    cv2.imwrite(f"result{i}.png", final)
