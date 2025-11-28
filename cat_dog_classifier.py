import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# 1. Descarga de datos y paths
# -----------------------------

URL = 'https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip'

path_to_zip = tf.keras.utils.get_file(
    'cats_and_dogs.zip',
    origin=URL,
    extract=True
)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# Obtener número de archivos para chequeo e hiperparámetros
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

print("total_train:", total_train)
print("total_val:", total_val)
print("total_test:", total_test)

# -----------------------------
# 2. Celda 3 – Generadores base
# -----------------------------

# ImageDataGenerator con sólo rescale
train_image_generator = ImageDataGenerator(rescale=1.0 / 255.)
validation_image_generator = ImageDataGenerator(rescale=1.0 / 255.)
test_image_generator = ImageDataGenerator(rescale=1.0 / 255.)

# Generadores a partir de las carpetas
train_data_gen = train_image_generator.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary'
)

val_data_gen = validation_image_generator.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary'
)

# Para el test, hay una carpeta "test" sin subcarpetas; en la notebook original
# resuelven esto creando un subdirectorio "test/test" o usando classes=['test'].
# Aquí replicamos lo que se usa en muchas soluciones FCC. :contentReference[oaicite:1]{index=1}
#
# Creamos un "dummy" subdirectorio si no existe.
test_wrapper_dir = os.path.join(PATH, 'test_wrapper')
test_subdir = os.path.join(test_wrapper_dir, 'test')

if not os.path.exists(test_wrapper_dir):
    os.makedirs(test_wrapper_dir)
if not os.path.exists(test_subdir):
    os.makedirs(test_subdir)

# Copiamos (o enlazamos) los archivos del test_dir original al subdir test_wrapper/test
for filename in os.listdir(test_dir):
    src = os.path.join(test_dir, filename)
    dst = os.path.join(test_subdir, filename)
    if not os.path.exists(dst):
        # Copia ligera (en muchos sistemas será rápida)
        try:
            os.link(src, dst)
        except OSError:
            # Si no se pueden crear hard links, copiar normal
            import shutil
            shutil.copy2(src, dst)

test_data_gen = test_image_generator.flow_from_directory(
    directory=test_wrapper_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,          # batch_size=1 para mantener sencillo el orden
    class_mode='binary',
    shuffle=False
)

print("\nGeneradores creados correctamente.")

# -----------------------------
# 3. Función plotImages (Celda 4)
# -----------------------------

def plotImages(images_arr, probabilities=None):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5, 5 * len(images_arr)))
    if len(images_arr) == 1:
        axes = [axes]

    for i, img in enumerate(images_arr):
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        if probabilities is not None:
            prob = probabilities[i]
            label = f"Dog: {prob*100:.2f}%  /  Cat: {(1-prob)*100:.2f}%"
            ax.set_title(label)
    plt.tight_layout()
    plt.show()

# Probar que anda (toma un batch del generador de train)
sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])

# -----------------------------
# 4. Celda 5 – Data augmentation
# -----------------------------

train_image_generator_aug = ImageDataGenerator(
    rescale=1.0 / 255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data_gen_aug = train_image_generator_aug.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary'
)

# Visualizar una imagen aumentada varias veces
augmented_images = []
sample_training_images, _ = next(train_data_gen_aug)
for i in range(5):
    augmented_images.append(sample_training_images[i])

plotImages(augmented_images)

# -----------------------------
# 5. Celda 7 – Definir modelo CNN
# -----------------------------

model = Sequential([
    Conv2D(32, (3, 3), activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  # salida binaria: dog/cat
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 6. Celda 8 – Entrenamiento
# -----------------------------

steps_per_epoch = total_train // batch_size
validation_steps = total_val // batch_size

history = model.fit(
    train_data_gen_aug,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=validation_steps
)

# -----------------------------
# 7. Celda 9 – Curvas de loss/accuracy
# -----------------------------

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

# -----------------------------
# 8. Celdas 10 y 11 – Predicción y test
# -----------------------------

# Probabilidad de "dog" para cada imagen del test
raw_probs = model.predict(test_data_gen)
# raw_probs tiene shape (n_imágenes, 1), lo aplastamos:
probabilities = raw_probs[:, 0]

# para mostrar algunas imágenes con sus probabilidades
test_images_batch, _ = next(test_data_gen)
# ojo: next(test_data_gen) da sólo el primer batch (con batch_size=1)
plotImages(test_images_batch, probabilities[:len(test_images_batch)])

# Ahora calculamos el porcentaje de aciertos igual que la notebook.
# En la notebook original hay un vector 'answers' con 0/1 correctos para cada imagen,
# que viene incluido en el propio notebook de FCC. :contentReference[oaicite:2]{index=2}
#
# Aquí vamos a replicar eso de forma manual:
# - Los nombres de archivo 1.jpg..25.jpg son gatos (0)
# - Los nombres de archivo 26.jpg..50.jpg son perros (1)
# (Este patrón es el que usan varias soluciones públicas de FCC.)

answers = []
filenames = sorted(os.listdir(test_dir), key=lambda x: int(os.path.splitext(x)[0]))
for fname in filenames:
    idx = int(os.path.splitext(fname)[0])
    if idx <= 25:
        answers.append(0)  # cat
    else:
        answers.append(1)  # dog

answers = np.array(answers)

# Reordenar probabilities en el mismo orden por nombre de archivo
# test_data_gen.filenames tiene algo como ['test/1.jpg', 'test/2.jpg', ...]
order = np.argsort([
    int(os.path.splitext(os.path.basename(f))[0])
    for f in test_data_gen.filenames
])
prob_sorted = probabilities[order]

predicted_labels = (prob_sorted >= 0.5).astype(int)

correct = np.sum(predicted_labels == answers)
percentage_identified = (correct / len(answers)) * 100

passed_challenge = percentage_identified >= 63.0

print(
    f"Your model correctly identified {round(percentage_identified, 2)}% "
    f"of the images of cats and dogs."
)

if passed_challenge:
    print("You passed the challenge!")
else:
    print(
        "You haven't passed yet. Your model should identify at least 63% "
        "of the images. Keep trying. You will get it!"
    )
