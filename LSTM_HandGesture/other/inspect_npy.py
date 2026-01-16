import numpy as np
import os
import sys

# === KONFIGURASI ===
# Ganti dengan nama folder gestur yang mau kamu intip
TARGET_CLASS = 'close_to_one' 
DATA_PATH = os.path.join('Keypoints_Data', TARGET_CLASS)

def inspect_data():
    # 1. Cek apakah folder ada
    if not os.path.exists(DATA_PATH):
        print(f"❌ Folder {DATA_PATH} tidak ditemukan!")
        return

    # 2. Ambil file pertama yang ketemu
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.npy')]
    if not files:
        print("❌ Tidak ada file .npy di folder ini.")
        return

    sample_file = files[0] # Ambil file pertama
    full_path = os.path.join(DATA_PATH, sample_file)

    print(f"🔍 Memeriksa file: {full_path}")
    
    # 3. Load Data
    data = np.load(full_path)

    # 4. Tampilkan Info Dasar
    print("-" * 30)
    print(f"Bentuk Data (Shape): {data.shape}")
    print(f"Tipe Data (Dtype): {data.dtype}")
    print("-" * 30)
    
    # 5. Tampilkan FULL DATA
    # Opsi ini memaksa NumPy mencetak semua angka tanpa disingkat (...)
    np.set_printoptions(threshold=sys.maxsize, linewidth=200, suppress=True)
    
    print("ISI DATA LENGKAP:")
    print(data)
    
    print("-" * 30)
    print("PENJELASAN UNTUK PENGUJI:")
    rows, cols = data.shape
    print(f"1. Matriks ini memiliki {rows} baris.")
    print(f"   -> Artinya: Video sampel ini terdiri dari {rows} frame.")
    print(f"2. Matriks ini memiliki {cols} kolom.")
    print(f"   -> Artinya: Setiap frame memiliki {cols} titik data.")
    print("   -> (21 Landmark Tangan) x (3 Koordinat x,y,z) = 63 Fitur.")

if __name__ == "__main__":
    inspect_data()