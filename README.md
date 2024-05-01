# Detección de Niveles Jugables con CNNs

## Resumen

Se desarrolló un algoritmo de redes neuronales que utiliza un conjunto de datos con diferentes imágenes de distintos niveles del videojuego Zelda. La máquina debe aprender a reconocer si el nivel de la imagen es jugable o no jugable. La forma de reconocer si un nivel es jugable se basa en los siguientes criterios:

1. El personaje de Link se encuentra en el nivel
2. Hay al menos una llave para que Link pueda acceder a la puerta
3. Hay al menos una puerta a la que Link pueda avanzar

Si no se cumple alguno de estos puntos, el nivel se clasificaría como no jugable.

Cabe mencionar que en las imágenes de los niveles, además de los objetos como Link, las llaves y las puertas, también se encuentran enemigos. Hay diferentes enemigos del juego con diferentes formas y colores. Si bien estos personajes no afectan en la decisión de si un nivel es jugable o no, es importante que el modelo reconozca solo aquellas características que sí importan. Estos enemigos podrían ser un estorbo visual para el aprendizaje del modelo y es algo a considerar en el desarrollo.

## Dataset

El dataset original fue recuperado de Kaggle con el título de **Zelda Game Levels**. Este dataset cuenta con 2048 archivos:

- 1024 con imágenes de niveles jugables (playable)
- 1024 con imágenes de niveles no jugables (unplayable)

Al recuperar el conjunto de datos, solo se encontraba dividido en 2 carpetas: playable y unplayable, representando las 2 clasificaciones a determinar.

Dataset: https://www.kaggle.com/datasets/adizafar/zelda-game-levels/data

## Preprocesamiento

Debido a que el conjunto de datos no se encontraba dividido en carpetas para entrenar, validar y probar, dividí las imágenes en estas 3 categorías para que mi modelo pueda utilizarlas después. Esta división se hizo de la siguiente manera:

- 70% para entrenamiento (1434 imágenes en total: 717 playable, 717 unplayable)
- 20% para test (408 imágenes en total: 204 playable, 204 unplayable)
- 10% para validación (206 imágenes en total: 103 playable, 103 unplayable)

Las imágenes que se usarán se encuentran en la carpeta llamada `dataset`. En este mismo se encuentran las imágenes divididas en subcarpetas tituladas `train`, `test` y `val` para entrenar, probar y validar el modelo. Dentro de cada uno de estos fólders, se vuelven a dividir en los niveles jugables y no jugables (carpetas `playable` y `unplayable`).

Asimismo, ya dentro de cada carpeta, todo el dataset pasó por técnicas de escalamiento y preprocesamiento como lo fue:

- Redimensionar todas las imágenes para que tengan la misma dimensión, ya que variaba en cada una.
- Recortar las imágenes para que solo tome en cuenta la parte que nos interesa, remover el marco de la imagen que no afecta a la decisión
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

Al momento de correrlo la primera vez, no se tenían las gráficas que ahora se tienen para la segunda versión, por lo que decidí correrlo una vez más con las nuevas gráficas. Al realizarlo con el nuevo código donde ya no se utiliza el generador de dataset, pude darme cuenta que los resultados fueron peor que antes y no eran correctos. A continuación los resultados de esta primera versión con la adición de las gráficas y correcciones en la generación de los valores finales.

#### Accuracy and Loss
| Accuracy | Loss |
|--- | --- |
|![alt text](/images/1_AccuracyTrain.png) | ![alt text](/images/1_LossTrain.png) |

#### Confusion Matrix
| Train | Test | Validation |
|--- | --- | --- |
|![alt text](/images/1_TrainConfusionMatrix.png) | ![alt text](/images/1_TestConfusionMatrix.png) | ![alt text](/images/1_ValidationConfusionMatrix.png) |


Fue entonces donde noté que mi modelo estaba realizando mucho underfitting y los resultados, como comentado antes, no eran para nada los esperados y la arquitectura no concordaba con lo que necesitaba mi modelo.


## Segunda versión

### Algoritmo

En la segunda versión hubo más cambios debido a que era necesario por el underfitting que tuvo previamente y la manera en la que se comportaba todo el modelo. Las capas utilizadas son las mismas en su mayoría pero con diferentes hiperparámetros, más repeticiones de capas y diferente orden. En este segundo modelo se utiliza Dropout para controlar el overfitting del modelo debido a las mejoras hechas.

#### Cambios

Los cambios hechos en esta segunda versión fueron los siguientes:

- Cambio en la arquitectura basado en el paper [1]
  - Arquitectura más profunda: Se añadieron más capas convolucionales que permiten a la red aprender caracteristicas más complejas y abstractas de las imágenes. Debido a que las imágenes contienen mucho detalle de todos los objetos y personajes presentes en ella, se añadieron más capas.
  - Más filtros: Se pusieron más filtros dentro de las capas que ayudan a captar más variedad de características.
  - Dropout: Se añade una capa de Dropout que ayuda a regular la red para que no haga overfitting con esta arquitectura más compleja.
- No se utiliza test datagen para el cálculo de la matriz de confusión, ahora se realiza manualmente importando el modelo y prediciendo cada imagen en el dataset
- Se añadieron más gráficas para su análisis

#### Arquitectura final
![alt text](/images/2_arquitectura.png)

### Resultados

Los resultados en la versión 2 fueron mejores que en la primera versión. Los valores en train y test incrementaron más, pero lo más destacable fue que la matriz de confusión ahora se mantiene siempre igual y en ella se puede notar que las predicciones realizadas son más atinadas a las verdaderas. Si bien el modelo sigue teniendo un poco de problema en identificar los niveles jugables como tal, en la gran mayoría de imágenes logra una clasificación como se debe. A continuación están las gráficas de este modelo:

**Puntuación F1 de train:** 0.9623

**Puntuación F1 de test:** 0.9382


#### Accuracy and Loss
| Accuracy | Loss |
|--- | --- |
|![alt text](/images/2_AccuracyTrain.png) | ![alt text](/images/2_LossTrain.png) |

#### Confusion Matrix and Values
| Train | Test | Validation |
|--- | --- | --- |
|![alt text](/images/2_TrainConfusionMatrix.png) | ![alt text](/images/2_TestConfusionMatrix.png) | ![alt text](/images/2_ValidationConfusionMatrix.png) |
|![alt text](/images/2_TrainValues.png) | ![alt text](/images/2_TestValues.png) | ![alt text](/images/2_ValidationValues.png) |

## Ejecución
Este nuevo modelo se encuentra terminado en el archivo `model2.py`. Asimismo, se puede ver el resultado dado en el archivo `model.ipynb`.

Si se desea correr algunas queries específicas para poner a prueba este modelo refinado se puede ejecutar el archivo `loadModel.py`. Este mismo archivo importa imágenes que no se encontraban en el dataset de la carpeta `/queries`. En el caso de querer introducir una nueva imagen se puede agregar en esta carpeta y modificar el código dentro de `loadModel.py` que importa el archivo deseado para realizar la predicción una vez que se ejecuta el código.

Por otro lado, en caso de querer probar un modelo para la generación de las matrices de confusión de train, test y validation y los valores importantes de ello, se puede ejecutar el código `plots.py`. Este archivo realiza las matrices de confusión e imprime los valores de cada categoría dentro de la terminal. Para correrlo es necesario tener un archivo .keras del modelo a probar. Se puede utilizar el archivo `pau.keras` para probar la versión más reciente del modelo.


## Conclusión
Después del análisis realizado de la primera versión y haciendo la investigación y ajustes necesarios para la segunda versión, se pudo ver un incremento en la precisión del modelo que satisface la predicción de las imágenes de diferentes niveles de Zelda en su gran mayoría. Si bien el modelo aún podría mejorar y puede ser refinado, cumple su propósito y se realizó una gran mejora en comparación a la primera versión. La segunda arquitectura es más compleja y poderosa que la primera y fue por ello que el aprendizaje fue más preciso y logró mejor precisión en la tarea de clasificación al terminar.


## Fuentes

- [1] Kunakornvong, P., & Asriny, D. M. (2019, June). Apple image classification using convolutional neural network. In 34th International Technology Conference Circuits/Systems, Computing Communication. https://www.researchgate.net/profile/Pichate-Kunakornvong/publication/339683856_Apple_image_classification_using_Convolutional_Neural_Network/links/5e5fa1d1a6fdccbeba19e6f3/Apple-image-classification-using-Convolutional-Neural-Network.pdf

- [2] Khor, W., Chen, Y.K., Roberts, M. et al. Automated detection and classification of concealed objects using infrared thermography and convolutional neural networks. Sci Rep 14, 8353 (2024). https://doi.org/10.1038/s41598-024-56636-8

- [3] Li, Y., Sycara, K., & Iyer, R. (2018). Object-sensitive deep reinforcement learning. arXiv preprint arXiv:1809.06064. https://arxiv.org/pdf/1809.06064.pdf

- [4] B. van Oostendorp, “Object Detection for Reinforcement Learning Agents”, Syst. Theor. Control Comput. J., vol. 3, no. 2, pp. 9–14, Dec. 2023, doi: 10.52846/stccj.2023.3.2.51.

- [5] Jung M, Yang H, Min K. "Improving Deep Object Detection Algorithms for Game Scenes". Electronics. 2021; 10(20):2527. https://doi.org/10.3390/electronics10202527