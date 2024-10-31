import cv2
import tensorflow as tf
import numpy as np
import json

# Load model VGG16 yang sudah dilatih
model = tf.keras.models.load_model('face_recognition_model_vgg16.keras')

# Load label dari file JSON
with open('label_map.json', 'r') as f:
    labels = json.load(f)

# Inisialisasi Video Capture
camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    check, frame = video.read()
    if not check:
        break  # Berhenti jika frame tidak berhasil diambil

    wajah = faceDeteksi.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in wajah:
        # Ekstraksi wajah dan ubah ukurannya agar sesuai dengan input model VGG16
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = np.expand_dims(face_img, axis=0) / 255.0  # Normalisasi

        # Prediksi wajah
        prediction = model.predict(face_img)
        confidence = np.max(prediction[0])  # Ambil confidence tertinggi

        if confidence > 0.8:  # Threshold confidence
            id = str(np.argmax(prediction[0]) + 1)  # Memulai ID dari 1
            label = labels.get(id, "Unknown")
        else:
            label = "Unknown"

        # Tampilkan prediksi dengan kotak hijau dan teks label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(label), (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1)
    if key == ord('a'):
        break

video.release()
cv2.destroyAllWindows()
