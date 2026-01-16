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

# Helper untuk membuat nama command latex aman (hapus spasi, jadi CamelCase)
# Contoh: 'D1 ON' -> 'DOneOn'
LABEL_TO_CMD = {
    'D1 ON': 'DOneOn',   'D1 OFF': 'DOneOff',
    'D2 ON': 'DTwoOn',   'D2 OFF': 'DTwoOff',
    'D3 ON': 'DThreeOn', 'D3 OFF': 'DThreeOff',
    'D4 ON': 'DFourOn',  'D4 OFF': 'DFourOff'
}

# Direktori tempat menyimpan hasil gambar
# Pastikan path ini sesuai dengan struktur folder Anda
OUTPUT_DIR = 'C:/zafaa/kuliah/SEMESTER7/PRATA/BukuTATekkomLatex/gambar/bab-4/confusion_matrix_pengujian'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path untuk file .tex yang digenerate
LATEX_OUTPUT_DIR = 'C:/zafaa/kuliah/SEMESTER7/PRATA/BukuTATekkomLatex/data' 
LATEX_FILE_PATH = os.path.join(LATEX_OUTPUT_DIR, 'akurasi_pengujian_all.tex')

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
    # cbar=True: Menampilkan color bar
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
# FUNGSI BARU: HITUNG AKURASI & GENERATE LATEX
# ==========================================
def hitung_akurasi_dan_generate_latex(data_matrix, scenario_prefix):
    """
    Menghitung akurasi per kelas dari CM dan membuat command LaTeX.
    Rumus Akurasi Kelas i = Matriks[i][i] / Sum(Baris i)
    """
    latex_commands = []
    
    # 1. Ambil diagonal (True Positives)
    true_positives = np.diag(data_matrix)
    # 2. Hitung total sampel per kelas (Jumlah per baris)
    total_per_class = np.sum(data_matrix, axis=1)
    
    accuracies = []
    for i, label in enumerate(CLASS_LABELS):
        # Hindari pembagian dengan nol jika total sampel kelas itu 0
        if total_per_class[i] > 0:
            acc = (true_positives[i] / total_per_class[i]) * 100
        else:
            acc = 0.0
        accuracies.append(acc)
        
        # Buat nama command, misal: \AccJarakDekatDOneOn
        cmd_name = f"Acc{scenario_prefix}{LABEL_TO_CMD[label]}"
        # Format ke 1 angka di belakang koma
        acc_str = f"{acc:.1f}"
        latex_commands.append(f"\\newcommand{{\\{cmd_name}}}{{{acc_str}}}")

    # Hitung rata-rata total
    avg_acc = np.mean(accuracies) if accuracies else 0.0
    avg_cmd_name = f"Acc{scenario_prefix}Avg"
    latex_commands.append(f"\\newcommand{{\\{avg_cmd_name}}}{{{avg_acc:.1f}}}")
    
    return "\n".join(latex_commands)

# ==========================================
# INPUT DATA MANUAL DI SINI
# ==========================================

if __name__ == "__main__":

    # ===================================================
    # BAGIAN 1: DEFINISI DATA PER SKENARIO
    # ===================================================
    
    # ---------------------------------------------------
    # SKENARIO 1: VARIASI JARAK
    # ---------------------------------------------------
    # 30 CM
    cm_data_jarak_30 = np.array([
        [15, 2,  1, 0,  0, 0,  0, 0], # Aktual: D1 ON
        [3, 14,  0, 1,  0, 0,  0, 0], # Aktual: D1 OFF
        [1, 0,  16, 3,  0, 0,  0, 0], # Aktual: D2 ON
        [0, 0,  2, 18,  0, 0,  0, 0], # Aktual: D2 OFF
        [0, 0,  0, 0,  12, 5,  1, 0], # Aktual: D3 ON
        [0, 0,  0, 0,  4, 16,  0, 0], # Aktual: D3 OFF
        [0, 0,  0, 0,  1, 0,  14, 3], # Aktual: D4 ON
        [0, 1,  0, 0,  0, 0,  2, 17], # Aktual: D4 OFF
    ])
    generate_manual_cm(cm_data_jarak_30, "Confusion Matrix - Jarak 30 cm", "jarak_30cm")

    # 50 CM
    cm_data_jarak_50 = np.array([
        [19, 1,  0, 0,  0, 0,  0, 0],
        [0, 20,  0, 0,  0, 0,  0, 0],
        [0, 0,  18, 2,  0, 0,  0, 0],
        [0, 0,  0, 20,  0, 0,  0, 0],
        [0, 0,  0, 0,  19, 1,  0, 0],
        [0, 0,  0, 0,  1, 19,  0, 0],
        [0, 0,  0, 0,  0, 0,  20, 0],
        [0, 0,  0, 0,  0, 0,  0, 20],
    ])
    generate_manual_cm(cm_data_jarak_50, "Confusion Matrix - Jarak 50 cm", "jarak_50cm")
    
    # 80 CM
    cm_data_jarak_80 = np.array([
        [19, 1,  0, 0,  0, 0,  0, 0],
        [0, 20,  0, 0,  0, 0,  0, 0],
        [0, 0,  18, 2,  0, 0,  0, 0],
        [0, 0,  0, 20,  0, 0,  0, 0],
        [0, 0,  0, 0,  19, 1,  0, 0],
        [0, 0,  0, 0,  1, 19,  0, 0],
        [0, 0,  0, 0,  0, 0,  20, 0],
        [0, 0,  0, 0,  0, 0,  0, 20],
    ])
    generate_manual_cm(cm_data_jarak_80, "Confusion Matrix - Jarak 80 cm", "jarak_80cm")
    
    
    # ---------------------------------------------------
    # SKENARIO 2: VARIASI PENCAHAYAAN
    # ---------------------------------------------------
    # REDUP
    cm_data_cahaya_redup = np.array([
        [15, 2,  1, 0,  0, 0,  0, 0],
        [3, 14,  0, 1,  0, 0,  0, 0],
        [1, 0,  16, 3,  0, 0,  0, 0],
        [0, 0,  2, 18,  0, 0,  0, 0],
        [0, 0,  0, 0,  12, 5,  1, 0],
        [0, 0,  0, 0,  4, 16,  0, 0],
        [0, 0,  0, 0,  1, 0,  14, 3],
        [0, 1,  0, 0,  0, 0,  2, 17],
    ])
    generate_manual_cm(cm_data_cahaya_redup, "Confusion Matrix - Cahaya Redup", "cahaya_redup")

    # SEDANG
    cm_data_cahaya_sedang = np.array([
        [19, 1,  0, 0,  0, 0,  0, 0],
        [0, 20,  0, 0,  0, 0,  0, 0],
        [0, 0,  18, 2,  0, 0,  0, 0],
        [0, 0,  0, 20,  0, 0,  0, 0],
        [0, 0,  0, 0,  19, 1,  0, 0],
        [0, 0,  0, 0,  1, 19,  0, 0],
        [0, 0,  0, 0,  0, 0,  20, 0],
        [0, 0,  0, 0,  0, 0,  0, 20],
    ])
    generate_manual_cm(cm_data_cahaya_sedang, "Confusion Matrix - Cahaya Sedang", "cahaya_sedang")
    
    # TERANG
    cm_data_cahaya_terang = np.array([
        [19, 1,  0, 0,  0, 0,  0, 0],
        [0, 20,  0, 0,  0, 0,  0, 0],
        [0, 0,  18, 2,  0, 0,  0, 0],
        [0, 0,  0, 20,  0, 0,  0, 0],
        [0, 0,  0, 0,  19, 1,  0, 0],
        [0, 0,  0, 0,  1, 19,  0, 0],
        [0, 0,  0, 0,  0, 0,  20, 0],
        [0, 0,  0, 0,  0, 0,  0, 20],
    ])
    generate_manual_cm(cm_data_cahaya_terang, "Confusion Matrix - Cahaya Terang", "cahaya_terang")
    
    
    # ---------------------------------------------------
    # SKENARIO 3: VARIASI RESOLUSI
    # ---------------------------------------------------
    # 360p
    cm_data_resolusi_360p = np.array([
        [15, 2,  1, 0,  0, 0,  0, 0],
        [3, 14,  0, 1,  0, 0,  0, 0],
        [1, 0,  16, 3,  0, 0,  0, 0],
        [0, 0,  2, 18,  0, 0,  0, 0],
        [0, 0,  0, 0,  12, 5,  1, 0],
        [0, 0,  0, 0,  4, 16,  0, 0],
        [0, 0,  0, 0,  1, 0,  14, 3],
        [0, 1,  0, 0,  0, 0,  2, 17],
    ])
    generate_manual_cm(cm_data_resolusi_360p, "Confusion Matrix - Resolusi 360p", "resolusi_360p")

    # 480p
    cm_data_resolusi_480p = np.array([
        [19, 1,  0, 0,  0, 0,  0, 0],
        [0, 20,  0, 0,  0, 0,  0, 0],
        [0, 0,  18, 2,  0, 0,  0, 0],
        [0, 0,  0, 20,  0, 0,  0, 0],
        [0, 0,  0, 0,  19, 1,  0, 0],
        [0, 0,  0, 0,  1, 19,  0, 0],
        [0, 0,  0, 0,  0, 0,  20, 0],
        [0, 0,  0, 0,  0, 0,  0, 20],
    ])
    generate_manual_cm(cm_data_resolusi_480p, "Confusion Matrix - Resolusi 480p", "resolusi_480p")
    
    # 720p
    cm_data_resolusi_720p = np.array([
        [19, 1,  0, 0,  0, 0,  0, 0],
        [0, 20,  0, 0,  0, 0,  0, 0],
        [0, 0,  18, 2,  0, 0,  0, 0],
        [0, 0,  0, 20,  0, 0,  0, 0],
        [0, 0,  0, 0,  19, 1,  0, 0],
        [0, 0,  0, 0,  1, 19,  0, 0],
        [0, 0,  0, 0,  0, 0,  20, 0],
        [0, 0,  0, 0,  0, 0,  0, 20],
    ])
    generate_manual_cm(cm_data_resolusi_720p, "Confusion Matrix - Resolusi 720p", "resolusi_720p")

    # ===================================================
    # BAGIAN 2: MENGHITUNG TOTAL CONFUSION MATRIX (OTOMATIS)
    # ===================================================
    print("\n--- Menghitung Total Confusion Matrix Gabungan ---")

    # 1. Kumpulkan semua variabel matriks di atas ke dalam satu list
    semua_matriks_skenario = [
        cm_data_jarak_30,
        cm_data_jarak_50,
        cm_data_jarak_80,
        cm_data_cahaya_redup,
        cm_data_cahaya_sedang,
        cm_data_cahaya_terang,
        cm_data_resolusi_360p,
        cm_data_resolusi_480p,
        cm_data_resolusi_720p
    ]

    # 2. Jumlahkan semua matriks dalam list tersebut
    # np.sum dengan axis=0 akan menjumlahkan elemen di posisi yang sama dari semua matriks
    cm_data_total_gabungan = np.sum(semua_matriks_skenario, axis=0)

    # 3. Generate gambar untuk total gabungan
    generate_manual_cm(
        data_matrix=cm_data_total_gabungan,
        title="Confusion Matrix Per Gestur",
        filename_suffix="total_gabungan"
    )
    
    print("\n--- Semua gambar Confusion Matrix (termasuk total gabungan) selesai dibuat ---")
    
    # ===================================================
    # BAGIAN BARU: KAMUS DATA UTAMA & GENERATE LATEX
    # ===================================================
    
    # 1. Daftarkan semua matriks dan beri nama prefix untuk command LaTeX-nya
    # Prefix ini harus unik dan deskriptif
    data_skenario = {
        "JarakDekat": cm_data_jarak_30,
        "JarakIdeal": cm_data_jarak_50,
        "JarakJauh":  cm_data_jarak_80,
        "CahayaRedup":  cm_data_cahaya_redup,
        "CahayaSedang": cm_data_cahaya_sedang,
        "CahayaTerang": cm_data_cahaya_terang,
        "ResRendah": cm_data_resolusi_360p,
        "ResSedang": cm_data_resolusi_480p,
        "ResTinggi": cm_data_resolusi_720p,
        "TotalGabungan": cm_data_total_gabungan
    }

    print("--- Memulai Proses Generasi ---")
    full_latex_content = "% File DIGENERATE OTOMATIS OLEH PYTHON. JANGAN DIEDIT MANUAL.\n"
    full_latex_content += "% Waktu generate: " + np.datetime64('now').astype(str) + "\n\n"

    # 2. Loop semua skenario, generate gambar, dan hitung akurasi
    for prefix, matrix in data_skenario.items():
        print(f"Memproses: {prefix}...")
        
        # Hitung akurasi dan buat command latex
        latex_block = f"% --- Data untuk Skenario: {prefix} ---\n"
        latex_block += hitung_akurasi_dan_generate_latex(matrix, prefix)
        latex_block += "\n\n"
        full_latex_content += latex_block

    # 3. Tulis ke file .tex
    try:
        with open(LATEX_FILE_PATH, "w") as f:
            f.write(full_latex_content)
        print(f"\n✅ SUKSES! File data LaTeX berhasil dibuat di:\n{LATEX_FILE_PATH}")
        print("Sekarang Anda bisa menggunakan command seperti \\AccJarakDekatDOneOn di dokumen LaTeX Anda.")
    except Exception as e:
        print(f"\n❌ ERROR saat menulis file LaTeX: {e}")