import numpy as np
import os
import sys

# === KONFIGURASI ===
# Ganti dengan nama folder gestur yang mau kamu intip
TARGET_CLASS = 'close_to_one' 
# Pastikan path ini benar di komputer Anda
DATA_PATH = os.path.join('C:/zafaa/kuliah/SEMESTER7/PRATA/code_gesture/LSTM_HandGesture/Keypoints_Data', TARGET_CLASS)

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
    
    # 5. Tampilkan FULL DATA (RAW) dengan presisi 4 angka
    # precision=4: Membatasi output jadi 4 angka belakang koma
    # suppress=True: Mencegah notasi ilmiah (e.g., 1e-5)
    np.set_printoptions(threshold=sys.maxsize, linewidth=150, suppress=True, precision=4)
    
    print("ISI DATA LENGKAP (RAW MATRIX - 4 DESIMAL):")
    print(data)
    
    # ==============================================================================
    # BAGIAN BARU: DETAIL PENJELASAN KOORDINAT (CONTOH FRAME PERTAMA)
    # ==============================================================================
    print("\n" + "="*60)
    print("DETAIL STRUKTUR DATA (CONTOH: HANYA FRAME KE-0)")
    print("="*60)
    print("Penjelasan: 63 kolom di atas adalah 21 titik landmark x 3 koordinat (X,Y,Z) secara berurutan.")
    print("-" * 80)
    # Header Tabel
    print(f"{'Landmark ID':<12} | {'Indeks Array':<14} | {'Koordinat X':<12} {'Koordinat Y':<12} {'Koordinat Z':<12}")
    print("-" * 80)

    # Ambil hanya frame pertama (baris ke-0) untuk contoh
    first_frame_data = data[0]
    num_landmarks = 21

    # Loop sebanyak 21 kali (untuk 21 titik tangan)
    for i in range(num_landmarks):
        # Setiap landmark memakan 3 indeks (0,1,2 lalu 3,4,5 dst.)
        base_idx = i * 3
        
        # Ambil nilai x, y, z dan format jadi string 4 desimal
        # :<8 artinya rata kiri dengan lebar 8 karakter agar rapi
        x_val = f"{first_frame_data[base_idx]:.4f}"
        y_val = f"{first_frame_data[base_idx+1]:.4f}"
        z_val = f"{first_frame_data[base_idx+2]:.4f}"
        
        # String untuk rentang indeks (misal: [ 0- 2])
        idx_range = f"[{base_idx:2d}-{base_idx+2:2d}]"
        
        print(f"Titik ke-{i:<5} | {idx_range:<14} | X={x_val:<10} Y={y_val:<10} Z={z_val:<10}")

    print("-" * 80)

    # 6. Penjelasan Penutup
    print("\n" + "-" * 30)
    print("PENJELASAN RINGKAS UNTUK PENGUJI:")
    rows, cols = data.shape
    print(f"1. Matriks ini memiliki {rows} baris.")
    print(f"   -> Artinya: Video sampel ini terdiri dari {rows} frame berurutan (dimensi waktu).")
    print(f"2. Matriks ini memiliki {cols} kolom.")
    print(f"   -> Artinya: Setiap frame memiliki {cols} fitur numerik (dimensi spasial).")
    print("   -> Didapat dari: (21 Titik Landmark Tangan) x (3 Koordinat X, Y, Z) = 63 Fitur.")

if __name__ == "__main__":
    inspect_data()