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
- Recortar las imagenes para que solo tome en cuenta la parte que nos interesa, remover el marco de la imagen que no afectaba a la decisión
- Normalizar las imágenes

Todo esto se realizó con la ayuda de unos scripts que se encuentran en el archivo `helper.py`, mismo que se puede volver a ocupar en caso de requerirlo en el algoritmo.

Es notable mencionar que las imágenes dentro del folder `dataset` ya se encuentra preprocesadas con lo ya mencionado anteriormente. Sin embargo, dentro del folder `original_dataset` se pueden observar todo el dataset original sin ninguna modificación.


## Estado de arte

Tras una investigación sobre el estado de arte hubo varias investigaciones que llamaron mi atención para implementar mi modelo. Pero la que mayormente destacó fue una que habla sobre el aprendizaje profundo por refuerzo sensible a objetos [1]. El O-DRL mejora el aprendizaje produnfo por refuerzo al incorporar información sobre objetos en el proceso de aprendizaje. Esto se logra al agregar más canales a la representación del estado del agente que codifican la presencia y ubicación de los objetos detectados en la imagen. Así el agente puede diferenciar entre objetos importantes y evitar objetos dañinos.

Debido a que en mi dataset y proyecto, las cosas a identificar esenciales son los objetos en los niveles, como lo serían las llaves, puertas y el personaje de Link. Este paper va muy bien para aplicarlo a mi modelo y que sea más fácil en identificar los objetos. Ya que esta técnica ayuda mucha en computer vision para la detección de objetos. En mi caso creo que puedo aplicar la arquitectura que se plantea en el que puedo aplicar hacer el gradiente en mi número de capas. Ir disminuyendo el tamaño de la capa conforme va avanzando. Asimismo, creo que sería buena práctica realizar el mapa de saliencia tal y como lo mencionan en el artículo. Creo que esta implementación puede ayudar a entender cómo está realizando y analizando las imágenes en mi algoritmo.


### Fuentes
[1] Li, Y., Sycara, K., & Iyer, R. (2018). Object-sensitive deep reinforcement learning. arXiv preprint arXiv:1809.06064. https://arxiv.org/pdf/1809.06064.pdf

[2] B. van Oostendorp, “Object Detection for Reinforcement Learning Agents”, Syst. Theor. Control Comput. J., vol. 3, no. 2, pp. 9–14, Dec. 2023, doi: 10.52846/stccj.2023.3.2.51.

[3] Jung M, Yang H, Min K. "Improving Deep Object Detection Algorithms for Game Scenes". Electronics. 2021; 10(20):2527. https://doi.org/10.3390/electronics10202527


El dataset original fue recuperado de: https://www.kaggle.com/datasets/adizafar/zelda-game-levels/data
