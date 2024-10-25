import cv2

# Menginisialisasi webcam dan detektor wajah
camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Meminta input ID pengguna
id = input('Masukan id : ')
a = 0

while True:
    a += 1
    # Membaca frame dari video
    check, frame = video.read()
    
    # Konversi ke grayscale
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Tampilkan gambar grayscale asli
    cv2.imshow("Original Gray Image", abu)
    
    # Terapkan histogram equalization
    abu_equalized = cv2.equalizeHist(abu)

        # Terapkan Gaussian filter untuk mengurangi noise
    abu_filtered = cv2.GaussianBlur(abu_equalized, (5, 5), 0)
    
    # Tampilkan gambar setelah histogram equalization
    cv2.imshow("Equalized Image", abu_equalized)
    
    # Deteksi wajah pada gambar yang telah di-equalize
    wajah = faceDeteksi.detectMultiScale(abu_equalized, 1.3, 5)
    
    # Menampilkan wajah yang terdeteksi dan menyimpannya
    for (x, y, w, h) in wajah:
        # Simpan gambar wajah yang telah di-equalize
        cv2.imwrite('DataSet/User.' + str(id) + '.' + str(a) + '.jpg', abu_equalized[y:y + h, x:x + w])
        # Membuat kotak hijau di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Tampilkan frame dengan kotak deteksi wajah
    cv2.imshow("Face Recognition", frame)
    
    # Tombol untuk keluar (tekan tombol apa saja untuk keluar)
    key = cv2.waitKey(1)
    
    # Berhenti setelah menyimpan 30 gambar
    if a > 129:
        break

# Melepaskan webcam dan menutup semua jendela
video.release()
cv2.destroyAllWindows()
