#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

base_dir = r'C:\Users\DELL\OneDrive\Desktop\aac2'
IMAGE_SIZE = 224
BATCH_SIZE = 64

# ImageDataGenerator setup for training and testing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
)

train_datagen = train_datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset="training"
)

test_datagen = test_datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation"
)

# CNN model setup
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=64, padding='same', strides=2, kernel_size=3, activation='relu', input_shape=(224, 224, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32, padding='same', strides=2, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(10, activation='softmax'))
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

his = cnn.fit(train_datagen, steps_per_epoch=16, epochs=50, verbose=1, validation_data=test_datagen, validation_steps=16)

# Class labels for prediction
class_labels = ['banana', 'coconut', 'coleus', 'curryleaf', 'goldenpothos', 'mint', 'neem', 'peepal', 'rose', 'turmeric']

def predict_image():
    global label_result, canvas, label_heading, label_confidence

    file_path = filedialog.askopenfilename()
    if file_path:
        new_image = plt.imread(file_path)
        new_image = tf.image.resize(new_image, (224, 224))
        new_image = np.expand_dims(new_image, axis=0)

        predictions = cnn.predict(new_image)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        confidence = np.max(predictions) * 100

        label_result.config(text=f"Predicted class: {predicted_class}\nPredicted label: {predicted_label}", font=("Arial", 14))
        #label_confidence.config(text=f"Confidence: {confidence:.2f}%", font=("Arial", 14))

        img = Image.open(file_path)
        img = img.resize((300, 300))  
        img = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

        label_result.place(relx=0.5, rely=0.8, anchor=tk.CENTER)
        label_confidence.place(relx=0.5, rely=0.85, anchor=tk.CENTER)

def set_custom_style():
    root.configure(bg='green')

    # Set background image for the root window
    background_image_path = r'C:\Users\DELL\Downloads\plant detection back.jpg'  # Replace with your image path
    background_image = Image.open(background_image_path)
    background_image = ImageTk.PhotoImage(background_image)
    background_label = tk.Label(root, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    background_label.image = background_image

# Tkinter window setup
root = tk.Tk()
root.title("Plant type detection")

set_custom_style()

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

btn_select_image = tk.Button(frame, text="Select Image", command=predict_image)
btn_select_image.pack(side=tk.LEFT)

canvas = tk.Canvas(root, width=300, height=300)
canvas.pack()

label_result = tk.Label(root, text="", font=("Arial", 12))
label_result.pack()

label_confidence = tk.Label(root, text="", font=("Arial", 12))
label_confidence.pack()

label_heading = tk.Label(root, text="PLANT PREDICTION DASHBOARD", font=("Arial", 20, "bold"), bg='green')
label_heading.pack()

root.mainloop()


# In[ ]:





# In[ ]:




