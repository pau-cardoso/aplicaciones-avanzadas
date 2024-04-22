# Aplicaciones Avanzadas
En este repositorio se encuentran los archivos de la materia **TC3002B: Desarrollo de aplicaciones avanzadas de ciencias computacionales**

En el fólder `M2/ZeldaLevels` se estará realizando un algoritmo de ML que tomará un dataset en donde están diferentes imágenes con diferentes niveles del videojuego Zelda. La máquina deberá aprender a reconocer si el nivel de la imagen es jugable o no jugable. La manera de reconocer si un nivel es jugable es porque se debe cumplir que:
1. El personaje de Link se encuentre en el nivel
2. Haya al menos una puerta a la que Link pueda avanzar
3. Haya al menos una llave para que Link pueda acceder a la puerta

De no cumplirse alguno de estos puntos, el nivel sería clasificado como no jugable.

Las imágenes que se estarán ocupando se encuentran ya divididas en fólders para entrenar y testear el algoritmo. Dentro de estos fólders, se vuelven a dividir en los niveles jugables y no jugables (playable y unplayable). Asimismo, ya en los fólders todo el dataset pasó por técnicas de escalamiento y preprocesado como lo fue:
- El redimensionar todas las imágenes para que estén con la misma dimensión ya que variaba en cada una.
- Recortar las imagenes para que solo tome en cuenta la parte que nos interesa, remover el marco de la imagen que no afectaba a la decisión
- Normalizar las imágenes

Todo esto se realizó con la ayuda de unos scripts que se encuentran en el archivo `helper.py`, mismo que se puede volver a ocupar en caso de requerirlo en el algoritmo.




El dataset original fue recuperado de: https://www.kaggle.com/datasets/adizafar/zelda-game-levels/data