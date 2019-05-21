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
        batch_y = self.imageY_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        X = []
        Y = []
        for file_name_y in batch_y:
            image = cv2.imread(file_name_y)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            # l,a,b = cv2.split(lab)
            l = lab[:,:,:1]
            ab = lab[:,:,1:]
            X.append(l)
            Y.append(ab)

        return np.array(X), np.array(Y)


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


NEW_PATH_X = './dataset/y'
NEW_PATH_Y = './dataset/z'
checkpoint_path = './model/'


batch_size = 64
paths_X = os.listdir(NEW_PATH_X)
paths_Y = os.listdir(NEW_PATH_Y)

complete_path_X = [os.path.join(NEW_PATH_X, image_path) for image_path in paths_X]
complete_path_Y = [os.path.join(NEW_PATH_Y, image_path) for image_path in paths_Y]

#num_training_samples = len(complete_path_X) - 950
num_training_samples = 128
num_epochs = 50
num_validation_samples = 64

X_train = complete_path_X[len(complete_path_X) - num_training_samples:len(complete_path_X)]
X_test = complete_path_X[:64]

y_train = complete_path_Y[len(complete_path_X) - num_training_samples:len(complete_path_X)]
y_test = complete_path_Y[:64]

# X = []
# Y = []
# for file_name_y in y_test:
#     image = cv2.imread(file_name_y)
#     lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
#     # l,a,b = cv2.split(lab)
#     l = lab[:,:,0]
#     ab = lab[:,:,1:]
#     X.append(l)
#     Y.append(ab)
# # for image in y_test:
# #     # print(image)
# #     img = cv2.imread(image)
# #     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
# #     l,a,b = cv2.split(lab)
# #     X.append([l])
# #     Y.append((a,b))
# X = np.array(X)
# Y = np.array(Y)
# print(X)
# print(X.shape)
# print("==================")
# print(Y)  
# print(Y.shape)


my_training_batch_generator = Generator(X_train, y_train, batch_size)
my_validation_batch_generator = Generator(X_test, y_test, batch_size)
checkpoint_path = "model/cp-{epoch:04d}.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 100-epochs.
    period=5)

# # tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model = create_model((256, 256, 1))
# print(model.summary())
# model.save_weights(checkpoint_path.format(epoch=0))
# model.fit_generator(image_a_b_gen(batch_size), steps_per_epoch=1, epochs=1000)

model.fit_generator(generator=my_training_batch_generator,
                    steps_per_epoch=(num_training_samples // batch_size) ,
                    epochs=num_epochs,
                    validation_data=my_validation_batch_generator,
                    validation_steps=(num_validation_samples // batch_size),
                    callbacks = [cp_callback],
                    verbose=1)



