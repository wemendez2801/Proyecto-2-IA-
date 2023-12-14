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
model.fit(x_train, y_train_one_hot, validation_data=(x_test, y_test_one_hot), epochs=3)

TF_SAVE_FILE = '/model/handwritten.h5'
model.save(TF_SAVE_FILE)
 """

#Graficación de resultados
""" plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Precisión del modelo')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.legend(['Entrenamiento', 'Validación'], loc='upper left')
plt.show() """

model = tf.keras.models.load_model('/model/handwritten.h5')

""" predictions = model.predict(x_test[:4])
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
    print("\n") """

#INTERFAZ
def preprocess_image(image):
    # Preprocesar la imagen antes de la predicción
    image = image.resize((28, 28))
    image = image.convert('L')  # Convertir a escala de grises
    #image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.invert(np.array(image))
    return image

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
        # Realizar la predicción y mostrar el resultado
        preprocessed_image = preprocess_image(self.image)
        prediction = predict_number(preprocessed_image)

        self.label_result.config(text=f"El número es: {prediction}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()