import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# Tentukan path ke folder DataSet
dataset_path = "D:/FaceRe/DataSet"

# Parameter untuk gambar
img_size = (128, 128)
batch_size = 32

# Menggunakan ImageDataGenerator untuk augmentasi data
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_gen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

valid_data = train_gen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Mendapatkan jumlah kelas dari generator
num_classes = train_data.num_classes

# Membangun model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(train_data, epochs=10, validation_data=valid_data)

# Simpan model yang sudah dilatih
model.save('face_recognition_model_cnn.h5')

# Membuat file label_map.json
label_map = train_data.class_indices  # Mendapatkan mapping label dari data generator
label_map = {str(int(v) + 1): k for k, v in label_map.items()}  # Mengatur ID mulai dari 1
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)

print("Label mapping berhasil disimpan ke label_map.json")
