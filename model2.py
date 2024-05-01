#### PACKAGE IMPORTS ####

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import numpy as np

# Rutas a los directorios de entrenamiento y prueba
train_dir = 'dataset/train'
test_dir = 'dataset/test'
val_dir = 'dataset/val'

# Crear generadores de imágenes para entrenamiento y validación
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
TARGET_SIZE = (88, 128)
EPOCHS = 25

# Cargar las imágenes de entrenamiento y prueba
train_generator = train_datagen.flow_from_directory(
  directory=train_dir,
  target_size=TARGET_SIZE,
  batch_size=BATCH_SIZE,
  class_mode='binary'
)

plt.imshow(train_generator[0][0][0])
plt.show()


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

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(88, 128, 3)),
    MaxPooling2D((2, 2), strides=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    Conv2D(512, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), strides=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Adding dropout for regularization to prevent overfitting
    Dense(1, activation='sigmoid')  # Using sigmoid activation for binary classification
])

# Compilar el modelo
model.compile(optimizer= 'adam',loss= 'binary_crossentropy', metrics = ['accuracy'] )

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    batch_size=10,
    validation_data=val_generator,
)

model.summary()

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_generator)
print('Precisión en el conjunto de prueba:', test_acc)

model.save("pau2.keras")


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

# Graficar loss
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


def check_manually(directory):
  file_predictions = []

  for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
      img_path = os.path.join(directory, filename)
      img = tf.keras.preprocessing.image.load_img(img_path, target_size=TARGET_SIZE)
      img_tensor = tf.keras.preprocessing.image.img_to_array(img)
      img_tensor = np.expand_dims(img_tensor, axis = 0)
      img_tensor /= 255.
      confidence = model.predict(img_tensor,  verbose = 0)
      file_predictions.append((confidence < 0.5).astype("int32"))

  return file_predictions

# Get the predictions for the test set
playable_predictions = check_manually(test_dir + '/playable')
unplayable_predictions = check_manually(test_dir + '/unplayable')

true_labels = np.array([1.0] * len(playable_predictions) + [0.0] * len(unplayable_predictions), dtype=np.float32)
predictions = np.concatenate([playable_predictions, unplayable_predictions]).flatten().astype(np.float32)
accuracy = np.mean(predictions == true_labels)
loss = tf.keras.losses.binary_crossentropy(true_labels, predictions).numpy().mean()
print("Accuracy:", accuracy)
print("Loss:", loss)

# Create a plot
plt.figure(figsize=(10, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(range(len(true_labels)), predictions, label='Predictions', color='blue')
plt.plot(range(len(true_labels)), true_labels, label='True Labels', color='red')
plt.title('Predictions vs True Labels')
plt.xlabel('Image Index')
plt.ylabel('Prediction / True Label')
plt.legend()
plt.show()

# playable = np.array(playable_predictions)
total_playable = len(playable_predictions)
positive_playable = np.sum(np.array(playable_predictions))
negative_playable = total_playable - positive_playable

total_unplayable = len(unplayable_predictions)
positive_unplayable = np.sum(np.array(unplayable_predictions))
negative_unplayable = total_unplayable - positive_unplayable

# Cálculo de métricas adicionales
TP = positive_playable
FP = negative_playable
TN = negative_unplayable
FN = positive_unplayable

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

# Plot confusion matrix
conf_matrix = [[TP, FP], [FN, TN]]
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
