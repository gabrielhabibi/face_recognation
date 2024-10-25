import cv2
import os
import numpy as np
from PIL import Image

# Membuat objek recognizer untuk metode LBPH (Local Binary Pattern Histogram)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Menggunakan Cascade Classifier untuk deteksi wajah
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Fungsi untuk mengambil gambar wajah dan label dari dataset
def getImagesWithLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []
    
    # Loop melalui setiap gambar di dataset
    for imagePath in imagePaths:
        # Membaca gambar dan mengubahnya menjadi grayscale
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        
        # Mendapatkan ID dari nama file
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        # Deteksi wajah dalam gambar
        faces = detector.detectMultiScale(imageNp)
        
        # Menyimpan wajah dan ID yang sesuai
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    
    return faceSamples, Ids

# Memanggil fungsi untuk mendapatkan gambar wajah dan label ID
faces, Ids = getImagesWithLabels('DataSet')

# Melatih recognizer menggunakan gambar wajah dan label
recognizer.train(faces, np.array(Ids))

# Menyimpan hasil training ke file XML
recognizer.save('DataSet/training.xml')
