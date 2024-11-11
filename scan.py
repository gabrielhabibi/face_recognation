import cv2
import os
import glob

# Path ke folder dataset
dataset_path = 'DataSet'

# Meminta input ID dan nama pengguna
user_id = input('Masukkan ID: ')
user_name = input('Masukkan Nama: ')

# Membuat subfolder berdasarkan ID dan Nama
user_folder = os.path.join(dataset_path, f"{user_id}_{user_name}")

# Cek apakah folder dengan ID sudah ada
if not os.path.exists(user_folder):
    os.makedirs(user_folder)
    print("Folder baru telah dibuat untuk ID ini.")
else:
    print("ID ini sudah terdaftar. Data rekam akan ditambahkan ke folder yang sudah ada.")

# Inisialisasi detektor wajah
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Fungsi untuk melakukan pengolahan citra dan menyimpan wajah yang di-crop
def process_and_crop_faces(image, file_name_prefix, save_folder, start_count=0, padding=0.2):
    # Terapkan histogram equalization per channel warna (RGB)
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Terapkan median filter untuk mengurangi noise
    image = cv2.medianBlur(image, 5) 

    # Deteksi wajah dalam gambar
    wajah = faceDeteksi.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    count = start_count

    for (x, y, w, h) in wajah:
        # Tambahkan padding di sekitar wajah
        x_pad = int(padding * w)
        y_pad = int(padding * h)

        # Pastikan padding tidak keluar dari batas gambar
        x_start = max(0, x - x_pad)
        y_start = max(0, y - y_pad)
        x_end = min(image.shape[1], x + w + x_pad)
        y_end = min(image.shape[0], y + h + y_pad)

        # Crop wajah dari gambar asli dengan padding
        cropped_face = image[y_start:y_end, x_start:x_end]
        
        # Resize ke (224, 224) agar konsisten dengan model
        cropped_face = cv2.resize(cropped_face, (224, 224))

        # Simpan gambar wajah yang sudah di-crop dengan nama yang sesuai
        file_name = f"{file_name_prefix}.{count}.jpg"
        file_path = os.path.join(save_folder, file_name)
        cv2.imwrite(file_path, cropped_face)
        print(f"Gambar wajah disimpan: {file_path}")

        count += 1
    return count

# Dapatkan indeks terakhir dari file yang ada di folder pengguna
existing_files = glob.glob(os.path.join(user_folder, f"User.{user_id}_{user_name}.*.jpg"))
index = len(existing_files)  # Mulai indeks berikutnya setelah file terakhir yang ada

# Memilih mode operasi: "kamera", "manual", atau "keduanya"
mode = input("Pilih mode (kamera/manual/keduanya): ").strip().lower()

if mode in ["kamera", "keduanya"]:
    # Bagian 1: Mendeteksi dan menyimpan wajah dari kamera
    camera = 0
    video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)

    print("Mengambil gambar dari kamera...")
    a = index
    while a < index + 30:  # Menyimpan hingga 30 gambar
        check, frame = video.read()

        # Proses cropping dan penyimpanan wajah dari kamera
        a = process_and_crop_faces(frame, f"User.{user_id}_{user_name}", user_folder, a)

        # Tampilkan frame dengan deteksi wajah
        for (x, y, w, h) in faceDeteksi.detectMultiScale(frame, 1.3, 5):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):  # Tekan 'q' untuk keluar lebih awal
            break

    video.release()
    cv2.destroyAllWindows()
    print("Pengambilan gambar dari kamera selesai.")

if mode in ["manual", "keduanya"]:
    # Bagian 2: Memproses gambar yang ditambahkan secara manual dalam subfolder pengguna
    print("Memproses gambar yang ditambahkan secara manual...")

    # Ambil semua file gambar di folder manual
    manual_images = glob.glob(os.path.join(user_folder, '*.jpg'))

    # Loop untuk setiap gambar di folder manual
    for image_path in manual_images:
        # Cek apakah nama file diawali dengan format "User.<ID>_<Nama>." untuk menghindari menghapus gambar dari kamera
        if not os.path.basename(image_path).startswith(f"User.{user_id}_{user_name}"):
            print(f"Membaca {image_path}")
            image = cv2.imread(image_path)

            if image is None:
                print(f"Gagal membaca {image_path}")
                continue

            # Terapkan pengolahan citra dan deteksi wajah
            index = process_and_crop_faces(image, f"User.{user_id}_{user_name}", user_folder, index)

            # Menghapus gambar asli setelah cropping selesai
            os.remove(image_path)
            print(f"Gambar asli dihapus: {image_path}")

    print("Proses gambar manual selesai.")

if mode not in ["kamera", "manual", "keduanya"]:
    print("Mode tidak dikenal. Harap pilih 'kamera', 'manual', atau 'keduanya'.")
