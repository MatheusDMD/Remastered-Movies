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
        # keras.layers.InputLayer(input_shape=(input_shape)),
        keras.layers.UpSampling2D(size=2, input_shape=input_shape, data_format=None, interpolation='nearest'),
        keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, padding='same'),
        # keras.layers.MaxPooling2D(pool_size = (2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation = tf.nn.relu, padding='same'),
        # keras.layers.MaxPooling2D(pool_size = (2, 2)),

        keras.layers.Conv2D(32, (3, 3), activation = tf.nn.relu, padding='same'),

        keras.layers.Conv2D(3, (3, 3), activation = tf.nn.relu, padding='same'),
        # keras.layers.MaxPooling2D(pool_size = (2, 2)),

        # keras.layers.Flatten(),

        # keras.layers.Dense(128, activation=tf.nn.relu),
        # keras.layers.Dense(64, activation=tf.nn.sigmoid)
    ])
    
    model.compile(optimizer='adam', 
              loss='mse',
              metrics=['accuracy'])
    
    return model


NEW_PATH_X = './dataset/x'
NEW_PATH_Y = './dataset/y'
checkpoint_path = './model/'


batch_size = 64
paths_X = os.listdir(NEW_PATH_X)
paths_Y = os.listdir(NEW_PATH_Y)

complete_path_X = [os.path.join(NEW_PATH_X, image_path) for image_path in paths_X]
complete_path_Y = [os.path.join(NEW_PATH_Y, image_path) for image_path in paths_Y]

num_training_samples = len(complete_path_X) - 200
num_epochs = 10
num_validation_samples = 200

X_train = complete_path_X[200:]
X_test = complete_path_X[:200]

y_train = complete_path_Y[200:]
y_test = complete_path_Y[:200]

# for image in X_train:
#     img = cv2.imread(image)
#     print(img.shape)


my_training_batch_generator = Generator(X_train, y_train, batch_size)
my_validation_batch_generator = Generator(X_test, y_test, batch_size)

# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model = create_model((128, 128, 3))
# print(model.summary())
model.fit_generator(generator=my_training_batch_generator,
                    steps_per_epoch=(num_training_samples // batch_size) ,
                    epochs=num_epochs,
                    validation_data=my_validation_batch_generator,
                    validation_steps=(num_validation_samples // batch_size))
                    # use_multiprocessing=True,
                    # workers=4,
                    # max_queue_size=32)
                    # callbacks=[tensorboard])


model.save_weights(checkpoint_path.format(epoch=0))