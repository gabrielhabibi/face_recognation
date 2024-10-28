import os
import shutil

# Tentukan path ke folder DataSet
dataset_path = 'DataSet'

# Membuat subfolder untuk setiap ID dan nama pengguna berdasarkan nama file
for file_name in os.listdir(dataset_path):
    if file_name.startswith("User"):
        # Mendapatkan ID dan nama dari nama file
        user_info = file_name.split('.')[1]  # Formatnya <ID>_<Nama>
        user_id, user_name = user_info.split('_')
        
        # Membuat nama folder sesuai ID dan Nama Pengguna
        subfolder_path = os.path.join(dataset_path, f"{user_id}_{user_name}")
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
        
        # Pindahkan file ke subfolder berdasarkan ID dan nama pengguna
        old_file_path = os.path.join(dataset_path, file_name)
        new_file_path = os.path.join(subfolder_path, file_name)
        shutil.move(old_file_path, new_file_path)
