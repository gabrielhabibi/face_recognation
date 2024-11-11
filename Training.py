import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import json

# Konfigurasi dataset
dataset_path = "DataSet"
test_dataset_path = "DataTest"
img_size = (224, 224)
batch_size = 32

# Augmentasi Data yang Disesuaikan
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

# Data training dan validasi
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

# Dapatkan jumlah kelas
num_classes = train_data.num_classes

# Model MobileNet sebagai base model
mobilenet = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Tambahkan layer klasifikasi
model = models.Sequential([
    mobilenet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Gunakan Adam dengan learning rate lebih rendah
optimizer = Adam(learning_rate=0.0001)

# Kompilasi model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callback untuk early stopping dan checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model_mobilenet.keras', monitor='val_accuracy', save_best_only=True)

# Training model
history = model.fit(train_data, epochs=100, validation_data=valid_data, callbacks=[early_stopping, checkpoint])

# Menampilkan hasil training dan validasi
print("Akurasi Training:", history.history['accuracy'])
print("Akurasi Validasi:", history.history['val_accuracy'])

# Simpan model terbaik setelah pelatihan
model.save('face_recognition_model_mobilenet.keras')

# Membuat file label_map.json untuk label kelas
label_map = train_data.class_indices
label_map = {str(int(v) + 1): k for k, v in label_map.items()}
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)

print("Label mapping berhasil disimpan ke label_map.json")

# ----- Evaluasi Model pada Data Testing -----
print("----- Evaluasi Model pada Data Testing -----")

# ImageDataGenerator untuk data testing tanpa augmentasi
test_gen = ImageDataGenerator(rescale=1.0 / 255)

# Data testing
test_data = test_gen.flow_from_directory(
    test_dataset_path,
    target_size=img_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluasi model pada data testing
test_loss, test_accuracy = model.evaluate(test_data)
print("Loss pada Data Testing:", test_loss)
print("Akurasi pada Data Testing:", test_accuracy)
