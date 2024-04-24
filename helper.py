import cv2
import os
import numpy as np
import tensorflow as tf

def load_images(data_dir):
    images = []
    labels = []
    categories =  ['playable', 'unplayable']

    for category in categories:
        print(data_dir, category)
        category_dir = os.path.join(data_dir, category)
        for image_file in os.listdir(category_dir):
            if image_file.endswith(".jpg"):
                image_path = os.path.join(category_dir, image_file)
                image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 88))
                image = tf.keras.preprocessing.image.img_to_array(image)
                images.append(image)

                if category == 'playable':
                    label = 1  # Etiqueta 1 para "playable"
                else:
                    label = 0  # Etiqueta 0 para "unplayable"

                labels.append(label)

    return images, labels


def resize_images():
    # Tamaños deseados
    target_width = 791
    target_height = 548

    # Directorio con las imagenes
    input_dir = "./ZeldaLevels/train/playable"
    output_dir = "resizing/train/playable"

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Lee la imagen
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Cambia el tamaño de la imagen
            resized_image = cv2.resize(image, (target_width, target_height))

            # Guarda la imagen con el nuevo tamaño
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, resized_image)

    print("All images resized and saved successfully!")


def normalize_images():
    # Directorio que contiene tus imágenes originales
    input_dir = "./ZeldaLevels/test/unplayable"
    output_dir = "normalize/test/unplayable"

    os.makedirs(output_dir, exist_ok=True)

    # Loop a través de cada imagen en el directorio de entrada
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Lee la imagen
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Normaliza la imagen dividiendo por 255
            normalized_image = image / 255.0

            # Guarda la imagen normalizada en el directorio de salida
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, (normalized_image * 255).astype(np.uint8))  # vuelve a escalar a 0-255 antes de guardarla

    print("Todas las imágenes han sido normalizadas y guardadas exitosamente!")


def crop_images():
    new_width = 675
    new_height = 433

    # Directorio que contiene las imágenes originales
    input_dir = "./ZeldaLevels/train/unplayable"
    output_dir = "crop/train/unplayable"

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Lee la imagen
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Obtener las dimensiones de la imagen
            height, width = image.shape[:2]

            # Calcular las coordenadas del rectángulo de recorte centrado
            start_x = max(0, int((width - new_width) / 2))
            start_y = max(0, int((height - new_height) / 2))
            end_x = min(width, start_x + new_width)
            end_y = min(height, start_y + new_height)

            # Recortar la región de interés centrada
            cropped_image = image[start_y:end_y, start_x:end_x]

            # Write the cropped image to the output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cropped_image)



# Escribir aquí las transformaciones que se deseen correr
# crop_images()