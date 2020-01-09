# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:56:14 2020

@author: Thomas.Harford
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

import confusion_matrix as cma



#import the data
(x_train, t_train) , (x_test, t_test) = datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0 # normalize pixel values
print('x_train shape: ', x_train.shape)

#hyperparameters
EPOCHS = 20


#build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

#compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

r = model.fit(x_train, t_train, validation_data=(x_test, t_test), epochs=EPOCHS)

#plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

#plot accuracy per iteration
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()

#evaluate the model
print(model.evaluate(x_test, t_test))

p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(t_test, p_test)
cma.plot_confusion_matrix(cm, list(range(10)))