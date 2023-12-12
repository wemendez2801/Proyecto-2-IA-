import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw

#Carga de datasets
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Normalización de datos
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
"""
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#Creación del modelo
model = Sequential()

#Capas
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Entrenamiento del modelo
model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot), epochs=3)

TF_SAVE_FILE = '/model/handwritten.h5'
model.save(TF_SAVE_FILE) """


#Graficación de resultados
""" plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show() """

model = tf.keras.models.load_model('/model/handwritten.h5')

predictions = model.predict(x_test[:4])
print(np.argmax(predictions, axis=1))
print(y_test[:4])

for i in range(4):
    image = x_test[i]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    print("Predicción: ", np.argmax(predictions[i]))
    print("Valor real: ", y_test[i])
    print("\n")