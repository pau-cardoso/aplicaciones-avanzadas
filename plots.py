# Este archivo ayuda a crear las graficas de un archivo del modelo (pau.keras)

import matplotlib.pyplot as plt
from tensorflow.keras import models, preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os

TARGET_SIZE = (88, 128)

# Rutas a los directorios de entrenamiento y prueba
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
VAL_DIR = 'dataset/val'

model = models.load_model("pau.keras")

def check_manually(directory): # Funcion para revisar manualmente las predicciones de las imagenes
  file_predictions = []

  for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
      img_path = os.path.join(directory, filename)
      img = preprocessing.image.load_img(img_path, target_size=TARGET_SIZE)
      img_tensor = preprocessing.image.img_to_array(img)
      img_tensor = np.expand_dims(img_tensor, axis = 0)
      img_tensor /= 255.
      confidence = model.predict(img_tensor,  verbose = 0)
      file_predictions.append((confidence < 0.5).astype("int32"))

  return file_predictions

def plot_confusion_matrix(directory, title):
  # Get the predictions for playable and unplayable images
  playable_predictions = check_manually(directory + '/playable')
  unplayable_predictions = check_manually(directory + '/unplayable')

  true_labels = np.array([1.0] * len(playable_predictions) + [0.0] * len(unplayable_predictions), dtype=np.float32)
  predictions = np.concatenate([playable_predictions, unplayable_predictions]).flatten().astype(np.float32)

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
  plt.title(title)
  plt.show()

print("\n------ Train ------")
plot_confusion_matrix(TRAIN_DIR, 'Train Confusion Matrix')
print("\n\n------ Test ------")
plot_confusion_matrix(TEST_DIR, 'Test Confusion Matrix')
print("\n\n------ Validation ------")
plot_confusion_matrix(VAL_DIR, 'Validation Confusion Matrix')