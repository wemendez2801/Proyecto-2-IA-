import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, TimeDistributed, LSTM, MaxPooling2D, LSTM, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from glob import glob
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
model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot), epochs=5) 

TF_SAVE_FILE = '/model/handwritten.h5'
model.save(TF_SAVE_FILE)  
"""
model = tf.keras.models.load_model('/model/handwritten.h5')

#INTERFAZ
def preprocess_image_cv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar suavizado para eliminar ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Aumentar el tamaño del kernel para un suavizado más grande

    # Aplicar umbral adaptativo con diferentes valores
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_imgs = []
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Descartar contornos muy pequeños
        if w * h > 100:
            digit = thresh[y:y+h, x:x+w]
            digit = cv2.resize(digit, (28, 28))
            digit_imgs.append(digit)
            bounding_boxes.append((x, digit))  # Se guarda solo la coordenada x junto al dígito

    # Ordenar los dígitos según su posición en el lienzo
    bounding_boxes.sort(key=lambda x: x[0])  # Ordenar por la coordenada x

    digit_imgs_sorted = [digit for _, digit in bounding_boxes]

    return digit_imgs_sorted

def predict_number(image):
    # Realizar la predicción utilizando el modelo cargado
    pred = model.predict(image)
    return np.argmax(pred)

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = Canvas(root, width=200, height=200, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=3)

        self.label_result = Label(root, text="El número es: ")
        self.label_result.grid(row=1, column=0, columnspan=3)

        self.button_predict = Button(root, text="Predecir", command=self.predict)
        self.button_predict.grid(row=2, column=0)

        self.button_clear = Button(root, text="Limpiar", command=self.clear_canvas)
        self.button_clear.grid(row=2, column=1)

        self.button_exit = Button(root, text="Salir", command=root.destroy)
        self.button_exit.grid(row=2, column=2)

        # Configurar eventos del mouse
        self.canvas.bind("<B1-Motion>", self.paint)

        # Inicializar variables
        self.old_x = None
        self.old_y = None
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<ButtonRelease-1>", self.reset_coords)

    def paint(self, event):
        # Dibujar en el canvas
        x, y = event.x, event.y
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, x, y, width=8, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.old_x, self.old_y, x, y], fill='black', width=8)

        self.old_x = x
        self.old_y = y
    
    def reset_coords(self, event):
        # Restablecer las coordenadas cuando el mouse se libera
        self.old_x = None
        self.old_y = None

    def clear_canvas(self):
        # Limpiar el canvas y la imagen
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="El número es: ")

    def predict(self):
        # Convertir la imagen del canvas a un formato compatible con OpenCV
        self.image.save("temp.png")
        image_cv = cv2.imread("temp.png")
    
        # Preprocesar la imagen usando OpenCV
        digit_imgs = preprocess_image_cv(image_cv)

        predictions = []
        for digit_img in digit_imgs:
            # Predecir cada dígito usando el modelo cargado
            digit_img = digit_img.reshape(1, 28, 28, 1)
            pred = model.predict(digit_img)
            prediction = np.argmax(pred)
            predictions.append(str(prediction))

        predicted_number = int(''.join(predictions))
        self.label_result.config(text=f"El número es: {predicted_number}")

        # Eliminar el archivo temporal
        os.remove("temp.png")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()