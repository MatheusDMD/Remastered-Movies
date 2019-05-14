import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence
from tensorflow import keras
from time import time
# from keras.callbacks import TensorBoard


class Generator(Sequence):

    def __init__(self, imageX_filenames, imageY_filenames, batch_size):
        self.imageX_filenames, self.imageY_filenames = imageX_filenames, imageY_filenames
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.imageX_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.imageX_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.imageY_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([cv2.imread(file_name_x) / 255.0 for file_name_x in batch_x]), np.array([cv2.imread(file_name_y) / 255.0 for file_name_y in batch_y])


def create_model(input_shape):
    model = keras.Sequential([

        keras.layers.Conv2D(64, (3,3), input_shape=input_shape, activation=tf.nn.relu, padding='same'),
        keras.layers.Conv2D(64,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(128,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(128,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(256,(3, 3), activation = tf.nn.relu, padding='same'),
        #keras.layers.Dropout(0.5), #Thanos
        keras.layers.Conv2D(256,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(256,(3, 3), activation = tf.nn.relu, padding='same'),
        #keras.layers.Dropout(0.25), #Half-Thanos
        keras.layers.Conv2D(128,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(64,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(32,(3, 3), activation = tf.nn.relu, padding='same'),
        keras.layers.Conv2D(3, (3, 3), activation = tf.nn.relu, padding='same'),
    ])
    
    model.compile(optimizer='adam', 
              loss='mse',
              metrics=['accuracy'])
    
    return model


NEW_PATH_X = '../dataset/y'
NEW_PATH_Y = '../dataset/z'
checkpoint_path = './model/'


batch_size = 64
paths_X = os.listdir(NEW_PATH_X)
paths_Y = os.listdir(NEW_PATH_Y)

complete_path_X = [os.path.join(NEW_PATH_X, image_path) for image_path in paths_X]
complete_path_Y = [os.path.join(NEW_PATH_Y, image_path) for image_path in paths_Y]

#num_training_samples = len(complete_path_X) - 950
num_training_samples = 1024
num_epochs = 2000
num_validation_samples = 128

X_train = complete_path_X[len(complete_path_X) - num_training_samples:len(complete_path_X)]
X_test = complete_path_X[:128]

y_train = complete_path_Y[len(complete_path_X) - num_training_samples:len(complete_path_X)]
y_test = complete_path_Y[:128]

# for image in y_test:
#     print(image)
#     img = cv2.imread(image)
#     item = img // 255
    # print(item)


my_training_batch_generator = Generator(X_train, y_train, batch_size)
my_validation_batch_generator = Generator(X_test, y_test, batch_size)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=10)

# # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model = create_model((256, 256, 3))
# print(model.summary())
model.fit_generator(generator=my_training_batch_generator,
                    steps_per_epoch=(num_training_samples // batch_size) ,
                    epochs=num_epochs,
                    validation_data=my_validation_batch_generator,
                    validation_steps=(num_validation_samples // batch_size),
                    callbacks = [cp_callback],
                    verbose=1)



# model.save_weights(checkpoint_path.format(epoch=0))
