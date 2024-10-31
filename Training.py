from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import os
import json

# Tentukan path ke folder DataSet
dataset_path = "DataSet"

# Parameter untuk gambar
img_size = (128, 128)
batch_size = 32

# Menggunakan ImageDataGenerator untuk augmentasi data
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

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

# Menggunakan VGG16 sebagai base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze layer pada base model, hanya unfreeze beberapa layer terakhir
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Menambahkan lapisan tambahan untuk klasifikasi
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),  # L2 Regularization
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping untuk mencegah overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Melatih model dengan EarlyStopping
history = model.fit(train_data, epochs=20, validation_data=valid_data, callbacks=[early_stopping])

# Menampilkan akurasi training dan validasi dari hasil training
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

print("Akurasi Data Training per Epoch:", train_accuracy)
print("Akurasi Data Validasi per Epoch:", val_accuracy)

# Simpan model yang sudah dilatih
model.save('face_recognition_model_vgg16.keras')

# Membuat file label_map.json
label_map = train_data.class_indices
label_map = {str(int(v) + 1): k for k, v in label_map.items()}
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)

print("Label mapping berhasil disimpan ke label_map.json")
