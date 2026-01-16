import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# KONFIGURASI LABEL KELAS
# ==========================================
CLASS_LABELS = [
    'D1 ON',  'D1 OFF',
    'D2 ON',  'D2 OFF',
    'D3 ON',  'D3 OFF',
    'D4 ON',  'D4 OFF'
]

# Direktori tempat menyimpan hasil gambar
OUTPUT_DIR = 'C:/zafaa/kuliah/SEMESTER7/PRATA/BukuTATekkomLatex/gambar/bab-4/confusion_matrix_pengujian'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# FUNGSI PEMBUAT GAMBAR (JANGAN DIUBAH)
# ==========================================
def generate_manual_cm(data_matrix, title, filename_suffix):
    """
    Fungsi untuk menghasilkan dan menyimpan gambar Confusion Matrix dari data manual.
    """
    # Validasi bentuk matriks
    expected_size = len(CLASS_LABELS)
    if data_matrix.shape != (expected_size, expected_size):
        print(f"ERROR: Ukuran matriks salah untuk {title}. Seharusnya {expected_size}x{expected_size}.")
        return

    plt.figure(figsize=(10, 8)) # Mengatur ukuran gambar output

    # Membuat Heatmap menggunakan Seaborn
    # annot=True: Menampilkan angka di dalam kotak
    # fmt='d': Format angka sebagai integer (bulat)
    # cmap='Blues': Skema warna (bisa diganti 'Reds', 'Greens', 'YlGnBu', dll.)
    # cbar=False: Menghilangkan color bar di samping (opsional, biar lebih bersih)
    sns.heatmap(data_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, 
                annot_kws={"size": 12}) # Ukuran font angka

    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('Label Aktual (Sebenarnya)', fontsize=12)
    plt.xlabel('Label Prediksi Sistem', fontsize=12)
    
    # Merapikan layout agar label tidak terpotong
    plt.tight_layout()

    # Menyimpan gambar
    filepath = os.path.join(OUTPUT_DIR, f'cm_{filename_suffix}.png')
    plt.savefig(filepath, dpi=300) # dpi=300 untuk kualitas cetak tinggi
    print(f"✅ Gambar berhasil disimpan: {filepath}")
    plt.close() # Menutup plot agar tidak menumpuk di memori

# ==========================================
# INPUT DATA MANUAL DI SINI
# ==========================================
# CARA MENGISI MATRIKS:
# - Setiap LIST di dalam array adalah SATU BARIS (Label Aktual).
# - Angka di dalam list adalah kolom (Label Prediksi).
#
# Contoh logika untuk baris pertama (D1 ON Aktual):
# [Jml D1 ON terdeteksi D1 ON, Jml D1 ON terdeteksi D1 OFF, Jml D1 ON terdeteksi D2 ON, ...]

if __name__ == "__main__":

    # ---------------------------------------------------
    # SKENARIO 1: VARIASI JARAK
    # ---------------------------------------------------
    # 30 CM
    # ---------------------------------------------------
    cm_data_jarak_30 = np.array([
        # Prediksi: D1ON, D1OFF, D2ON, D2OFF, D3ON, D3OFF, D4ON, D4OFF
        [15, 2,  1, 0,  0, 0,  0, 0], # Aktual: D1 ON
        [3, 14,  0, 1,  0, 0,  0, 0], # Aktual: D1 OFF
        [1, 0,  16, 3,  0, 0,  0, 0], # Aktual: D2 ON
        [0, 0,  2, 18,  0, 0,  0, 0], # Aktual: D2 OFF
        [0, 0,  0, 0,  12, 5,  1, 0], # Aktual: D3 ON
        [0, 0,  0, 0,  4, 16,  0, 0], # Aktual: D3 OFF
        [0, 0,  0, 0,  1, 0,  14, 3], # Aktual: D4 ON
        [0, 1,  0, 0,  0, 0,  2, 17], # Aktual: D4 OFF
    ])
    generate_manual_cm(
        data_matrix=cm_data_jarak_30,
        title="Confusion Matrix - Jarak 30 cm",
        filename_suffix="jarak_30cm"
    )

    # ---------------------------------------------------
    # 50 CM
    # ---------------------------------------------------
    cm_data_jarak_50 = np.array([
        # Prediksi: D1ON, D1OFF, D2ON, D2OFF, D3ON, D3OFF, D4ON, D4OFF
        [19, 1,  0, 0,  0, 0,  0, 0], # Aktual: D1 ON
        [0, 20,  0, 0,  0, 0,  0, 0], # Aktual: D1 OFF
        [0, 0,  18, 2,  0, 0,  0, 0], # Aktual: D2 ON
        [0, 0,  0, 20,  0, 0,  0, 0], # Aktual: D2 OFF
        [0, 0,  0, 0,  19, 1,  0, 0], # Aktual: D3 ON
        [0, 0,  0, 0,  1, 19,  0, 0], # Aktual: D3 OFF
        [0, 0,  0, 0,  0, 0,  20, 0], # Aktual: D4 ON
        [0, 0,  0, 0,  0, 0,  0, 20], # Aktual: D4 OFF
    ])
    generate_manual_cm(
        data_matrix=cm_data_jarak_50,
        title="Confusion Matrix - Jarak 50 cm",
        filename_suffix="jarak_50cm"
    )
    
    # ---------------------------------------------------
    # 80 CM
    # ---------------------------------------------------
    cm_data_jarak_80 = np.array([
        # Prediksi: D1ON, D1OFF, D2ON, D2OFF, D3ON, D3OFF, D4ON, D4OFF
        [19, 1,  0, 0,  0, 0,  0, 0], # Aktual: D1 ON
        [0, 20,  0, 0,  0, 0,  0, 0], # Aktual: D1 OFF
        [0, 0,  18, 2,  0, 0,  0, 0], # Aktual: D2 ON
        [0, 0,  0, 20,  0, 0,  0, 0], # Aktual: D2 OFF
        [0, 0,  0, 0,  19, 1,  0, 0], # Aktual: D3 ON
        [0, 0,  0, 0,  1, 19,  0, 0], # Aktual: D3 OFF
        [0, 0,  0, 0,  0, 0,  20, 0], # Aktual: D4 ON
        [0, 0,  0, 0,  0, 0,  0, 20], # Aktual: D4 OFF
    ])
    generate_manual_cm(
        data_matrix=cm_data_jarak_80,
        title="Confusion Matrix - Jarak 80 cm",
        filename_suffix="jarak_80cm"
    )
    
    
    # ---------------------------------------------------
    # SKENARIO 2: VARIASI PENCAHAYAAN
    # ---------------------------------------------------
    # XX LUX - REDUP
    # ---------------------------------------------------
    cm_data_cahaya_redup = np.array([
        # Prediksi: D1ON, D1OFF, D2ON, D2OFF, D3ON, D3OFF, D4ON, D4OFF
        [15, 2,  1, 0,  0, 0,  0, 0], # Aktual: D1 ON
        [3, 14,  0, 1,  0, 0,  0, 0], # Aktual: D1 OFF
        [1, 0,  16, 3,  0, 0,  0, 0], # Aktual: D2 ON
        [0, 0,  2, 18,  0, 0,  0, 0], # Aktual: D2 OFF
        [0, 0,  0, 0,  12, 5,  1, 0], # Aktual: D3 ON
        [0, 0,  0, 0,  4, 16,  0, 0], # Aktual: D3 OFF
        [0, 0,  0, 0,  1, 0,  14, 3], # Aktual: D4 ON
        [0, 1,  0, 0,  0, 0,  2, 17], # Aktual: D4 OFF
    ])
    generate_manual_cm(
        data_matrix=cm_data_cahaya_redup,
        title="Confusion Matrix - Cahaya Redup",
        filename_suffix="cahaya_redup"
    )

    # ---------------------------------------------------
    # XX LUX - SEDANG
    # ---------------------------------------------------
    cm_data_cahaya_sedang = np.array([
        # Prediksi: D1ON, D1OFF, D2ON, D2OFF, D3ON, D3OFF, D4ON, D4OFF
        [19, 1,  0, 0,  0, 0,  0, 0], # Aktual: D1 ON
        [0, 20,  0, 0,  0, 0,  0, 0], # Aktual: D1 OFF
        [0, 0,  18, 2,  0, 0,  0, 0], # Aktual: D2 ON
        [0, 0,  0, 20,  0, 0,  0, 0], # Aktual: D2 OFF
        [0, 0,  0, 0,  19, 1,  0, 0], # Aktual: D3 ON
        [0, 0,  0, 0,  1, 19,  0, 0], # Aktual: D3 OFF
        [0, 0,  0, 0,  0, 0,  20, 0], # Aktual: D4 ON
        [0, 0,  0, 0,  0, 0,  0, 20], # Aktual: D4 OFF
    ])
    generate_manual_cm(
        data_matrix=cm_data_cahaya_sedang,
        title="Confusion Matrix - Cahaya Sedang",
        filename_suffix="cahaya_sedang"
    )
    
    # ---------------------------------------------------
    # XX LUX - TERANG
    # ---------------------------------------------------
    cm_data_cahaya_terang = np.array([
        # Prediksi: D1ON, D1OFF, D2ON, D2OFF, D3ON, D3OFF, D4ON, D4OFF
        [19, 1,  0, 0,  0, 0,  0, 0], # Aktual: D1 ON
        [0, 20,  0, 0,  0, 0,  0, 0], # Aktual: D1 OFF
        [0, 0,  18, 2,  0, 0,  0, 0], # Aktual: D2 ON
        [0, 0,  0, 20,  0, 0,  0, 0], # Aktual: D2 OFF
        [0, 0,  0, 0,  19, 1,  0, 0], # Aktual: D3 ON
        [0, 0,  0, 0,  1, 19,  0, 0], # Aktual: D3 OFF
        [0, 0,  0, 0,  0, 0,  20, 0], # Aktual: D4 ON
        [0, 0,  0, 0,  0, 0,  0, 20], # Aktual: D4 OFF
    ])
    generate_manual_cm(
        data_matrix=cm_data_cahaya_terang,
        title="Confusion Matrix - Cahaya Terang",
        filename_suffix="cahaya_terang"
    )
    
    
    # ---------------------------------------------------
    # SKENARIO 3: VARIASI RESOLUSI
    # ---------------------------------------------------
    # 360p
    # ---------------------------------------------------
    cm_data_resolusi_360p = np.array([
        # Prediksi: D1ON, D1OFF, D2ON, D2OFF, D3ON, D3OFF, D4ON, D4OFF
        [15, 2,  1, 0,  0, 0,  0, 0], # Aktual: D1 ON
        [3, 14,  0, 1,  0, 0,  0, 0], # Aktual: D1 OFF
        [1, 0,  16, 3,  0, 0,  0, 0], # Aktual: D2 ON
        [0, 0,  2, 18,  0, 0,  0, 0], # Aktual: D2 OFF
        [0, 0,  0, 0,  12, 5,  1, 0], # Aktual: D3 ON
        [0, 0,  0, 0,  4, 16,  0, 0], # Aktual: D3 OFF
        [0, 0,  0, 0,  1, 0,  14, 3], # Aktual: D4 ON
        [0, 1,  0, 0,  0, 0,  2, 17], # Aktual: D4 OFF
    ])
    generate_manual_cm(
        data_matrix=cm_data_resolusi_360p,
        title="Confusion Matrix - Resolusi 360p",
        filename_suffix="resolusi_360p"
    )

    # ---------------------------------------------------
    # 480p
    # ---------------------------------------------------
    cm_data_resolusi_480p = np.array([
        # Prediksi: D1ON, D1OFF, D2ON, D2OFF, D3ON, D3OFF, D4ON, D4OFF
        [19, 1,  0, 0,  0, 0,  0, 0], # Aktual: D1 ON
        [0, 20,  0, 0,  0, 0,  0, 0], # Aktual: D1 OFF
        [0, 0,  18, 2,  0, 0,  0, 0], # Aktual: D2 ON
        [0, 0,  0, 20,  0, 0,  0, 0], # Aktual: D2 OFF
        [0, 0,  0, 0,  19, 1,  0, 0], # Aktual: D3 ON
        [0, 0,  0, 0,  1, 19,  0, 0], # Aktual: D3 OFF
        [0, 0,  0, 0,  0, 0,  20, 0], # Aktual: D4 ON
        [0, 0,  0, 0,  0, 0,  0, 20], # Aktual: D4 OFF
    ])
    generate_manual_cm(
        data_matrix=cm_data_resolusi_480p,
        title="Confusion Matrix - Resolusi 480p",
        filename_suffix="resolusi_480p"
    )
    
    # ---------------------------------------------------
    # 720p
    # ---------------------------------------------------
    cm_data_resolusi_720p = np.array([
        # Prediksi: D1ON, D1OFF, D2ON, D2OFF, D3ON, D3OFF, D4ON, D4OFF
        [19, 1,  0, 0,  0, 0,  0, 0], # Aktual: D1 ON
        [0, 20,  0, 0,  0, 0,  0, 0], # Aktual: D1 OFF
        [0, 0,  18, 2,  0, 0,  0, 0], # Aktual: D2 ON
        [0, 0,  0, 20,  0, 0,  0, 0], # Aktual: D2 OFF
        [0, 0,  0, 0,  19, 1,  0, 0], # Aktual: D3 ON
        [0, 0,  0, 0,  1, 19,  0, 0], # Aktual: D3 OFF
        [0, 0,  0, 0,  0, 0,  20, 0], # Aktual: D4 ON
        [0, 0,  0, 0,  0, 0,  0, 20], # Aktual: D4 OFF
    ])
    generate_manual_cm(
        data_matrix=cm_data_resolusi_720p,
        title="Confusion Matrix - Resolusi 720p",
        filename_suffix="resolusi_720p"
    )

    print("\n--- Semua gambar Confusion Matrix selesai dibuat ---")