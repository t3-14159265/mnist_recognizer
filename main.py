# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

#hyperparameters
EPOCHS = 5
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.Adam()
VALIDATION_SPLIT = 0.95
IMG_ROWS, IMG_COLS = 28, 28 #input dimensions

INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1) # GREYSCALE IMAGE WITH ONE 1 COLOUR CHANNEL
NB_CLASSES = 10 # digits 0 - 9 to be 

#define model
def build(input_shape, classes):
    model = models.Sequential()
    model.add(layers.Convolution2D(20, (5,5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Convolution2D(50, (5,5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(classes, activation="softmax"))
    return model


(x_train, y_train) , (x_test, y_test) = datasets.mnist.load_data()
#reshape
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

#normalize
x_train, x_test = x_train / 255.0, x_test / 255.0
#cast
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#one hot encode the targets
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)

# initialize the optimizer and model
model = build(input_shape=INPUT_SHAPE, classes=NB_CLASSES)
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=["accuracy"])
model.summary()

#use tensorboard
callbacks = [
    # write tensorboard logs to './logs' directory
    tf.keras.callbacks.TensorBoard(log_dir='.\logs')
    ]

#fit
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT, callbacks=callbacks)
score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

