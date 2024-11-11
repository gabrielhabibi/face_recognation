import cv2
import tensorflow as tf
import numpy as np
import json

# Load model yang sudah dilatih
model = tf.keras.models.load_model('face_recognition_model_mobilenet.keras')

# Load label dari file JSON
with open('label_map.json', 'r') as f:
    labels = json.load(f)

# Inisialisasi Video Capture
camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Inisialisasi variabel untuk lock frame
locked_label = "Unknown"
locked_confidence = 0
lock_frames = 0
threshold_confidence = 0.75  # Threshold confidence untuk menampilkan label
reset_lock_frames = 15  # Jumlah frame untuk reset lock

while True:
    check, frame = video.read()
    if not check:
        break
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    detected = False

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = np.expand_dims(face_img, axis=0) / 255.0

        prediction = model.predict(face_img)
        confidence = np.max(prediction[0])
        id = str(np.argmax(prediction[0]) + 1)
        current_label = labels.get(id, "Unknown")

        # Jika confidence tidak cukup, tetap tampilkan "Unknown"
        if confidence < threshold_confidence:
            locked_label = "Unknown"
            locked_confidence = confidence
            lock_frames = reset_lock_frames
        else:
            # Update label yang terkunci jika memenuhi syarat
            if (current_label != locked_label and confidence > locked_confidence * 1.05) or lock_frames <= 0:
                locked_label = current_label
                locked_confidence = confidence
                lock_frames = reset_lock_frames  # Reset lock frame
            detected = True

        # Tampilkan deteksi pada frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{locked_label} ({locked_confidence*100:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Kurangi lock_frames jika tidak ada wajah yang terdeteksi
    if not detected:
        lock_frames -= 1
        if lock_frames <= -reset_lock_frames:
            locked_label = "Unknown"
            locked_confidence = 0
            lock_frames = 0

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
