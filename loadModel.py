from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt

TARGET_SIZE = (88, 128)

model = models.load_model("pau.keras")

img_path = 'queries/test9.png'
labels = ["unplayable", "playable"]

plt.figure()
plt.imshow(image.load_img(img_path, target_size=TARGET_SIZE))

img = image.load_img(img_path,  target_size=TARGET_SIZE)
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor /= 255.
confidence = model.predict(img_tensor)
predict_class = (confidence < 0.5).astype("int32")
plt.title("Prediccion: " + labels[predict_class[0][0]])
print("Prediccion: " + labels[predict_class[0][0]])

plt.show()