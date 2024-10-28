import cv2
import tensorflow as tf
import numpy as np
import json

# Load model CNN yang sudah dilatih
model = tf.keras.models.load_model('face_recognition_model_cnn.h5')

# Load label dari file JSON
with open('label_map.json', 'r') as f:
    labels = json.load(f)

# Inisialisasi Video Capture
camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)
    for (x, y, w, h) in wajah:
        # Ekstraksi wajah dan ubah ukurannya agar sesuai dengan input model CNN
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = np.expand_dims(face_img, axis=0) / 255.0  # Normalisasi

        # Prediksi wajah
        prediction = model.predict(face_img)
        id = str(np.argmax(prediction[0]) + 1)  # Memulai ID dari 1
        label = labels.get(id, "Unknown")

        # Tampilkan prediksi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(label), (x + 40, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1)
    if key == ord('a'):
        break

video.release()
cv2.destroyAllWindows()
