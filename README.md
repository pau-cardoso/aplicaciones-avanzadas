# Detectando Niveles Jugables en Zelda

## Resumen

Se estará realizando un algoritmo de redes neuronales que tomará un dataset con diferentes imágenes de diferentes niveles del videojuego Zelda. La máquina deberá aprender a reconocer si el nivel de la imagen es jugable o no jugable. La manera de reconocer si un nivel es jugable es con base a los siguientes criterios:

1. El personaje de Link se encuentre en el nivel
2. Haya al menos una llave para que Link pueda acceder a la puerta
3. Haya al menos una puerta a la que Link pueda avanzar

De no cumplirse alguno de estos puntos, el nivel sería clasificado como no jugable.

Cabe mencionar que en las imágenes de los niveles además de los objetos como Link, las llaves y puertas, también se encuentran enemigos. Hay diferentes enemigos del juego con diferentes formas y colores. Si bien estos personajes no afectaran en la decisión de si un nivel es jugable o no, es importante que el modelo reconozca solo aquellas características que si importan. Estos enemigos podrían ser un estorbo visual para el aprendizaje del modelo y es algo a considerar en el desarrollo.

## Dataset

El dataset original fue recuperado de Kaggle con el título de **Zelda Game Levels**. Este dataset cuenta con 2048 archivos:

- 1024 con imágenes de niveles jugables (playable)
- 1024 con imágenes de niveles no jugables (unplayable)

Al recuperar el dataset únicamente se encontraba dividido en 2 fólders: playable y unplayable, representando las 2 clasificaciones a determinar.

Dataset: https://www.kaggle.com/datasets/adizafar/zelda-game-levels/data

## Preprocesamiento

Debido a que el dataset no se encontraba dividido en los fólder para entrenar, validar y testear, dividí las imágenes en estas 3 categorías para que mi modelo pueda utilizar posteriormente. Esta división se hizo de la siguiente manera:

- 70% para entrenamiento (1434 imágenes en total: 717 playable, 717 unplayable)
- 20% para test (408 imágenes en total: 204 playable, 204 unplayable)
- 10% para validación (206 imágenes en total: 103 playable, 103 unplayable)

Las imágenes que se estarán ocupando se encuentran en el fólder nombrado `dataset`. En este mismo se encuentran las imágenes divididas en subcarpetas tituladas `train`, `test` y `val` para entrenar, testear y validar el modelo. Dentro de cada uno de estos fólders, se vuelven a dividir en los niveles jugables y no jugables (carpetas `playable` y `unplayable`).

Asimismo, ya dentro de cada carpeta, todo el dataset pasó por técnicas de escalamiento y preprocesado como lo fue:

- El redimensionar todas las imágenes para que estén con la misma dimensión ya que variaba en cada una.
- Recortar las imagenes para que solo tome en cuenta la parte que nos interesa, remover el marco de la imagen que no afecta a la decisión
- Normalizar las imágenes

Todo esto se realizó con la ayuda de unos scripts que se encuentran en el archivo `helper.py`, mismo que se puede volver a ocupar en caso de requerirlo en el algoritmo.

Cabe mencionar que las imágenes dentro del folder `dataset` ya se encuentran preprocesadas con lo ya mencionado anteriormente. Sin embargo, dentro del folder `original_dataset` se puede observar el dataset original sin ninguna de las modificaciones mencionadas previamente.

## Estado de arte

Para el desarrollo del modelo, se realizó una investigación de diferentes papers, entre ellos los siguientes:

- [1] Kunakornvong, P., & Asriny, D. M. (2019, June). Apple image classification using convolutional neural network. In 34th International Technology Conference Circuits/Systems, Computing Communication. https://www.researchgate.net/profile/Pichate-Kunakornvong/publication/339683856_Apple_image_classification_using_Convolutional_Neural_Network/links/5e5fa1d1a6fdccbeba19e6f3/Apple-image-classification-using-Convolutional-Neural-Network.pdf

- [2] Khor, W., Chen, Y.K., Roberts, M. et al. Automated detection and classification of concealed objects using infrared thermography and convolutional neural networks. Sci Rep 14, 8353 (2024). https://doi.org/10.1038/s41598-024-56636-8

- [3] Li, Y., Sycara, K., & Iyer, R. (2018). Object-sensitive deep reinforcement learning. arXiv preprint arXiv:1809.06064. https://arxiv.org/pdf/1809.06064.pdf

- [4] B. van Oostendorp, “Object Detection for Reinforcement Learning Agents”, Syst. Theor. Control Comput. J., vol. 3, no. 2, pp. 9–14, Dec. 2023, doi: 10.52846/stccj.2023.3.2.51.

- [5] Jung M, Yang H, Min K. "Improving Deep Object Detection Algorithms for Game Scenes". Electronics. 2021; 10(20):2527. https://doi.org/10.3390/electronics10202527

Para la primera versión me basé mucho en el paper [2] debido a su enfoque en los objetos en las imágenes. Se basa en agregar más canales a la representación del estado para codificar la ubicación y la presencia de objetos en las imágenes. Sin embargo, este paper se base en una metodolofía de aprendizaje de refuerzo que no concuerda con el modelo deseado para este caso. Si bien me dio una base de inicio y la estructura de su arquitectura fue una ayuda, las razones y la base de ella fue incorrecta para el desarrollo de mi modelo.

Por esto mismo al realizar más investigaciones, encontré varios papers que se asimilaban más a mi problema y que ayudaban más a lo que necesitaba. El estado de arte principal en el que me basé fue el paper "Apple image classification using Convolutional Neural Network" [1]. En este paper, utilizan CNN para la detección de manzanas en buen estado o mal estado. La arquitectura de este modelo usa MLP (Multi-Layer Perceptron) que consta de varias capas de convolución, agrupamiento, eliminación de neuronas, aplanamiento y capas densas. El principal objetivo de este paper es clasificar las manzanas en 2 clases principales: buen estado o mal estado. Al revisar las imágenes del dataset utilizado, pude notar que la red neuronal debía analizar las características de las imágenes para poder determinar el resultado y es lo que necesitaba para mi problema de igual manera.

La arquitectura en este paper ayuda mucho al problema que deseo atacar, pues es muy importante que se analicen diferentes características de la imagen como el color, manchas, formas, etc.. Y con este mismo razonamiento, decidí utilizar la arquitectura que describen en el paper con ligeros cambios en la misma. Debido a que mis imágenes contienen mayor detalle y características a analizar, le añadí más capas siguiendo el mismo patrón utilizado para poder fortalecer más mi modelo. Esto me sirvió de base para el modelo presentado en este repositorio.

## Primera Versión

### Algoritmo

En mi primer versión del modelo, utilicé capas de Convolución, Pooling, Flatten y Dense, similar a la segunda versión. Sin embargo con un approach diferente ya que se realizaba de forma inversa. Me estaba basando en un paper de aprendizaje por refuerzo por lo que no era la mejor opción y por eso busqué otros papers que se asimilaban más a mi trabajo.

En esta primera versión se tenía la siguiente arquitectura:

- **Conv2D (64, (3, 3), activation='relu')**: Primera capa convolucional. Cuenta con 64 filtros de 3x3 pixeles con función de activación ReLU (Rectified Linear Unit).

- **MaxPooling2D((2, 2))**: Realiza un pooling con un filtro de 2x2 pixeles. Reduce la dimensionalidad de la representación y extrae las características dominantes.

- **Conv2D (32, (3, 3), activation='relu')**: Segunda capa convolucional. Tiene 32 filtros de 3x3 pixeles y con función de activación ReLU nuevamente.

- **MaxPooling2D((2, 2))**: Segunda capa de pooling. Reduce la dimensionalidad de la representación aprendida por la segunda capa convolucional.

- **Flatten()**: Convierte la salida de la capa anterior en un vector unidimensional para alimentar las capas totalmente conectadas.

- **Dense(64, activation='relu')**: Primera capa totalmente conectada con 64 neuronas y función de activación ReLU.

- **Dense(2, activation='softmax')**: Capa de salida de la red. Tiene dos neuronas porque la clasificación es binaria (para las categorías de playable y unplayable). La función de activación softmax hace que se pueda interpretar la salida como una probabilidad de pertenecer a una clase u otra.

Este tipo de arquitectura con capas convolucionales y pooling al inicio permiten capturar las características de las imágenes y luego con capas totalmente conectadas se procesa la información aprendida para realizar la clasificación.


### Resultados

En esta primera versión se tuvo un **accuracy de 0.8063 en entrenamiento** y uno de **0.8014705777168274 en testing**. Sin embargo, al sacar la matriz de confusión los resultados se puede notar que no está realizando las predicciones correctamente. Se nota que el algoritmo no identifica los niveles playable y los clasifica como unplayable. Esto fue indicativo de que mi modelo se encontraba en underfitting y era necesario meterle más capas en la arquitectura para que pudiera mejorar en el entrenamiento.

Otro problema con este algoritmo fue que, al estar usando data generators en mi algoritmo para entrenamiento, validación y testing. La matriz de confusión variaba cada vez que se corría debido a que la generación variaba cada vez y era difícil fiarse de ello.

En general esta arquitectura no fue un buen modelo para el problema a resolver. Hizo falta mejor investigación y una mejor arquitectura que correspondiera con la necesidad del modelo.

![alt text](/images/train_accuracy.png)
![alt text](/images/loss_accuracy.png)
![alt text](/images/test_confusion_matrix.png)

Otro comentario sobre esta primera versión fue que, al momento de correrlo la primera vez, no se tuvieron las gráficas como en la segunda versión. 


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
