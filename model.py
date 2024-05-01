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

BATCH_SIZE = 32
TARGET_SIZE = (128, 88)
EPOCHS = 15

def check_manually(directory):
  file_predictions = []

  for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
      img_path = os.path.join(directory, filename)
      img = filename.load_img(img_path,  target_size=TARGET_SIZE)
      img_tensor = filename.img_to_array(img)
      img_tensor = np.expand_dims(img_tensor, axis = 0)
      img_tensor /= 255.
      confidence = model.predict(img_tensor,  verbose = 0)
      file_predictions.append((confidence > 0.5).astype("int32"))

  return file_predictions

# Rutas a los directorios de entrenamiento y prueba
train_dir = 'dataset/train'
test_dir = 'dataset/test'
val_dir = 'dataset/val'

# Crear generadores de imágenes para entrenamiento y validación
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Cargar las imágenes de entrenamiento y prueba
train_generator = train_datagen.flow_from_directory(
  directory=train_dir,
  target_size=TARGET_SIZE,
  batch_size=BATCH_SIZE,
  class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
  directory=test_dir,
  target_size=TARGET_SIZE,
  batch_size=BATCH_SIZE,
  class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
  directory=val_dir,
  target_size=TARGET_SIZE,
  batch_size=BATCH_SIZE,
  class_mode='binary'
)

# Definir la arquitectura del modelo CNN
model = Sequential([
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.summary()

# Compilar el modelo
model.compile(optimizer= 'adam',loss= 'sparse_categorical_crossentropy',metrics = ['accuracy'] )
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    train_generator,
    # steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(1, len(acc)+1)

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_generator)
print('Precisión en el conjunto de prueba:', test_acc)

model.save("version1_pau.keras")

# Obtener accuracy y loss de entrenamiento y validación
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(1, len(acc)+1)

# Graficar accuracy
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure()

# Graficar loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_generator)

print('Accuracy en el conjunto de prueba:', test_acc)

# Get the predictions for the test set
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
# print("predicted_classes", predicted_classes)

# Get the true labels from the test generator
true_labels = test_generator.labels
# print("true_labels", true_labels)

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_classes)

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
