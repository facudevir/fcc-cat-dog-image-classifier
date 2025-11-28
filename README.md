# Cat and Dog Image Classifier – freeCodeCamp

Este repositorio contiene mi solución al proyecto **Cat and Dog Image Classifier** de freeCodeCamp (Machine Learning with Python).

El script principal es `cat_dog_classifier.py`. Hace lo siguiente:

1. Descarga y descomprime el dataset `cats_and_dogs` desde freeCodeCamp.
2. Crea generadores de imágenes (`ImageDataGenerator`) para train, validation y test.
3. Aplica data augmentation sobre las imágenes de entrenamiento para reducir overfitting.
4. Define una red neuronal convolucional (CNN) con TensorFlow / Keras.
5. Entrena el modelo y muestra las curvas de _loss_ y _accuracy_.
6. Predice sobre el set de test y calcula el porcentaje de acierto.
7. Imprime el mensaje final de si se pasó el desafío (>= 63% de accuracy).

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python cat_dog_classifier.py
