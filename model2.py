#### PACKAGE IMPORTS ####

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from helper import load_images
import random
import cv2

def shuffle_together(arr1, arr2):
    combined = list(zip(arr1, arr2))
    random.shuffle(combined)
    arr1[:], arr2[:] = zip(*combined)
    return arr1, arr2

def load_images(data_dir):
    images = []
    labels = []
    categories =  ['playable', 'unplayable']

    for category in categories:
        category_dir = os.path.join(data_dir, category)
        for image_file in os.listdir(category_dir):
            if image_file.endswith(".jpg"):
                image_path = os.path.join(category_dir, image_file)
                image = cv2.imread(image_path)
                image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 88))
                images.append(image)

                if category == 'playable':
                    label = 1  # Etiqueta 1 para "playable"
                else:
                    label = 0  # Etiqueta 0 para "unplayable"

                labels.append(label)

    images, labels = shuffle_together(images, labels)
    images = tf.convert_to_tensor(np.array(images))
    labels = tf.convert_to_tensor(np.array(labels))
    return images, labels


# Rutas a los directorios de entrenamiento y prueba
train_dir = 'dataset/train'
test_dir = 'dataset/test'
val_dir = 'dataset/val'

# Cargar las imágenes de entrenamiento y prueba
train_images, train_labels = load_images(train_dir)
test_images, test_labels = load_images(test_dir)
val_images, val_labels = load_images(val_dir)

BATCH_SIZE = 32
TARGET_SIZE = (128, 88)
EPOCHS = 30

# Definir la arquitectura del modelo CNN
model = Sequential([
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

# Compilar el modelo
model.compile(optimizer= 'adam',loss= 'binary_crossentropy', metrics = ['accuracy'] )

# Entrenar el modelo

history = model.fit(
    train_images,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(val_images, val_labels),
)

acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs,acc,'bo',label='train accuracy')
plt.title('train acc')
plt.legend()

plt.figure()

plt.plot(epochs,loss, 'bo', label ='training loss')
plt.title('train loss')
plt.legend()

plt.show()


# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Accuracy en el conjunto de prueba:', test_acc)

# Get the predictions for the test set
predictions = model.predict(test_images)
# Get the predicted classes for each image
predicted_classes = np.argmax(predictions, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# Cálculo de métricas adicionales
TP = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
TN = conf_matrix[1, 1]
FN = conf_matrix[1, 0]

true_positive_rate = TP / (TP + FN)
false_positive_rate = FP / (FP + TN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * ((precision * recall) / (precision + recall))

print('True positive rate:', true_positive_rate)
print('False positive rate:', false_positive_rate)
print('Precisión:', precision)
print('Sensibilidad (Recall):', recall)
print('Puntuación F1:', f1_score)