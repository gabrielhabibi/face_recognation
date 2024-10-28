import cv2

# Menginisialisasi webcam dan detektor wajah
camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Meminta input ID dan nama pengguna
user_id = input('Masukkan ID: ')
user_name = input('Masukkan Nama: ')
a = 0

while True:
    a += 1
    # Membaca frame dari video
    check, frame = video.read()
    
    # Konversi ke grayscale
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Terapkan histogram equalization
    abu_equalized = cv2.equalizeHist(abu)

    # Terapkan median filter untuk mengurangi noise
    abu_filtered = cv2.medianBlur(abu_equalized, 5)  # Ukuran kernel 5
    
    # Deteksi wajah pada gambar yang telah difilter
    wajah = faceDeteksi.detectMultiScale(abu_filtered, 1.3, 5)  # Gunakan gambar yang telah difilter

    # Menampilkan wajah yang terdeteksi dan menyimpannya
    for (x, y, w, h) in wajah:
        # Simpan gambar wajah yang telah di-equalize dengan format ID dan nama
        cv2.imwrite(f'DataSet/User.{user_id}_{user_name}.{a}.jpg', abu_filtered[y:y + h, x:x + w])
        # Membuat kotak hijau di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Tampilkan frame dengan kotak deteksi wajah
    cv2.imshow("Face Recognition", frame)
    
    # Tombol untuk keluar (tekan tombol apa saja untuk keluar)
    key = cv2.waitKey(1)
    
    # Berhenti setelah menyimpan 30 gambar
    if a > 29:
        break

# Melepaskan webcam dan menutup semua jendela
video.release()
cv2.destroyAllWindows()
