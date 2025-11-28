import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------
# 1. Descarga de datos y unzip
# -----------------------------

URL = 'https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip'

# Descargamos el zip SIN extraer
zip_path = tf.keras.utils.get_file(
    'cats_and_dogs.zip',
    origin=URL,
    extract=False
)

# Lo extraemos nosotros en el directorio actual del proyecto
extract_dir = os.path.join(os.getcwd(), 'cats_and_dogs')

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# PATH ahora es la carpeta que contiene 'train' y 'validation'
PATH = extract_dir

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

print("PATH:", PATH)
print("Contenido de PATH:", os.listdir(PATH))

# Contar imágenes de train y validation
def count_files(root):
    total = 0
    for r, d, files in os.walk(root):
        total += len(files)
    return total

total_train = count_files(train_dir) if os.path.exists(train_dir) else 0
total_val = count_files(validation_dir) if os.path.exists(validation_dir) else 0

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

print("total_train:", total_train)
print("total_val:", total_val)

if total_train == 0 or total_val == 0:
    raise RuntimeError(
        "No se encontraron imágenes en train o validation. "
        "Revisá la estructura extraída en 'cats_and_dogs/'."
    )

# -----------------------------
# 2. Celda 3 – Generadores base
# -----------------------------

train_image_generator = ImageDataGenerator(rescale=1.0 / 255.)
validation_image_generator = ImageDataGenerator(rescale=1.0 / 255.)

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

print("\nGeneradores de train y validation creados correctamente.")

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

# Visualizar algunas imágenes aumentadas
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

print(f"\nÚltima validation accuracy: {val_acc[-1] * 100:.2f}%")
