# Detectando Niveles Jugables en Zelda

## Resumen
En el archivo `model.ipynb` se estará realizando un algoritmo de ML que tomará un dataset en donde están diferentes imágenes con diferentes niveles del videojuego Zelda. La máquina deberá aprender a reconocer si el nivel de la imagen es jugable o no jugable. La manera de reconocer si un nivel es jugable es porque se debe cumplir que:
1. El personaje de Link se encuentre en el nivel
2. Haya al menos una puerta a la que Link pueda avanzar
3. Haya al menos una llave para que Link pueda acceder a la puerta

De no cumplirse alguno de estos puntos, el nivel sería clasificado como no jugable.

## Preprocesamiento

Las imágenes que se estarán ocupando se encuentran ya divididas en fólders para entrenar y testear el algoritmo. Dentro de estos fólders, se vuelven a dividir en los niveles jugables y no jugables (playable y unplayable). Asimismo, ya en los fólders todo el dataset pasó por técnicas de escalamiento y preprocesado como lo fue:
- El redimensionar todas las imágenes para que estén con la misma dimensión ya que variaba en cada una.
- Recortar las imagenes para que solo tome en cuenta la parte que nos interesa, remover el marco de la imagen que no afecta a la decisión
- Normalizar las imágenes

Todo esto se realizó con la ayuda de unos scripts que se encuentran en el archivo `helper.py`, mismo que se puede volver a ocupar en caso de requerirlo en el algoritmo.

Es notable mencionar que las imágenes dentro del folder `dataset` ya se encuentra preprocesadas con lo ya mencionado anteriormente. Sin embargo, dentro del folder `original_dataset` se pueden observar todo el dataset original sin ninguna modificación.


## Estado de arte
El modelo escogido fue basado en el paper "Apple image classification using Convolutional Neural Network" [1]. En este paper, utilizan CNN para la detección de manzanas en buen estado o mal estado. La arquitectura de este modelo usa MLP (Multi-Layer Perceptron) que consta de varias capas de convolución, agrupamiento, eliminación de neuronas, aplanamiento y capas densas. Debido a que el uso que le dan a esta arquitectura es muy parecido a mi problema en el sentido de que en este problema es muy importante que se analicen características de la imagen como el color, manchas, etc. fue un modelo que se alinea más a mi problemática. Utilicé la arquitectura de este paper para mi modelo, sin embargo, debido a que mis imágenes contienen mayor detalle y características, le añadí más capas siguiendo la misma teoría utilizada.

## Primera Versión

### Algoritmo
En mi primer versión del modelo, utilicé capas de Convolución, Pooling, Flatten y Dense, similar a la segunda versión. Sin embargo con un approach diferente ya que se realizaba de forma inversa. Me estaba basando en un paper de refuerzo por lo que no era la mejor opción y por eso busqué otros papers que se asimilaban más a mi trabajo.

![alt text](/images/1_arquitectura.png)

### Resultados
En esta primera versión se tuvo un **accuracy de 0.8063 en entrenamiento** y uno de **0.8014705777168274 en testing**. Sin embargo, al sacar la matriz de confusión los resultados se puede notar que no está realizando las predicciones correctamente. Se nota que el algoritmo no identifica los niveles playable y los clasifica como unplayable.

Otro problema con este algoritmo fue que, al estar usando data generators en mi algoritmo para entrenamiento, validación y testing. La matriz de confusión variaba cada vez que se corría debido a que la generación del dataset variaba cada vez y entonces era difícil fiarse de ello.

![alt text](/images/train_accuracy.png)
![alt text](/images/loss_accuracy.png)
![alt text](/images/test_confusion_matrix.png)


## Segunda versión

### Algoritmo
En la segunda versión el modelo utiliza las mismas capas de la primera versión pero de forma inversa y con más repeticiones. Asimismo utiliza funciones Dropout para controlar el overfitting del modelo. Cabe mencionar que las convoluciones se realizan con activación de ReLU, esto porque en el paper mencionado es el que utilizan debido a que en estudios previos se dieron cuenta que ReLU es más rápido que tanh en el entrenamiento y era una mejor opción para este tipo de problema.

### Cambios
En la segunda versión se realizaron varios cambios entre ellos:
- Cambio en la arquitectura
- Se añaden más repeticiones en las capas para incrementar la detección de las características
- No se utiliza test datagen para el cálculo de la matriz de confusión, ahora se realiza manualmente

![alt text](/images/2_arquitectura.png)

### Resultados
Los resultados en la versión 2 fueron mejores que en la primera versión. El accuracy en train y test incrementó un poco, pero la matriz de confusión ahora se mantiene siempre igual y en ella se puede notar que las predicciones realizadas son más atinadas a las verdaderas. Sigue teniendo un poco de problema en identificar los niveles jugables como tal, pues en esa parte es donde se equivoca bastante. Sin embargo las equivocaciones al intentar predecir que un nivel es no jugable son mínimas y lo hace muy bien.

**train accuracy:** 0.8635

**test accuracy:** 0.8333

#### Matriz de confusión
![alt text](/images/2_conf_matrix.png)




## Fuentes
[1] Kunakornvong, P., & Asriny, D. M. (2019, June). Apple image classification using convolutional neural network. In 34th International Technology Conference Circuits/Systems, Computing Communication. https://www.researchgate.net/profile/Pichate-Kunakornvong/publication/339683856_Apple_image_classification_using_Convolutional_Neural_Network/links/5e5fa1d1a6fdccbeba19e6f3/Apple-image-classification-using-Convolutional-Neural-Network.pdf
[2] Li, Y., Sycara, K., & Iyer, R. (2018). Object-sensitive deep reinforcement learning. arXiv preprint arXiv:1809.06064. https://arxiv.org/pdf/1809.06064.pdf

[3] B. van Oostendorp, “Object Detection for Reinforcement Learning Agents”, Syst. Theor. Control Comput. J., vol. 3, no. 2, pp. 9–14, Dec. 2023, doi: 10.52846/stccj.2023.3.2.51.

[4] Jung M, Yang H, Min K. "Improving Deep Object Detection Algorithms for Game Scenes". Electronics. 2021; 10(20):2527. https://doi.org/10.3390/electronics10202527


El dataset original fue recuperado de: https://www.kaggle.com/datasets/adizafar/zelda-game-levels/data
