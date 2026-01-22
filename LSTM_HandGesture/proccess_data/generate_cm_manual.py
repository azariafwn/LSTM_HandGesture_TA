import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd # Import Pandas wajib ada

# ==========================================
# KONFIGURASI LABEL KELAS
# ==========================================
# Urutan ini PENTING. Harus sama persis dengan yang ada di CSV (Target_Gesture & Last_Command)
CLASS_LABELS = [
    'D1 ON',  'D1 OFF',
    'D2 ON',  'D2 OFF',
    'D3 ON',  'D3 OFF',
    'D4 ON',  'D4 OFF'
]

# Helper untuk membuat nama command latex aman (hapus spasi, jadi CamelCase)
LABEL_TO_CMD = {
    'D1 ON': 'DOneOn',   'D1 OFF': 'DOneOff',
    'D2 ON': 'DTwoOn',   'D2 OFF': 'DTwoOff',
    'D3 ON': 'DThreeOn', 'D3 OFF': 'DThreeOff',
    'D4 ON': 'DFourOn',  'D4 OFF': 'DFourOff'
}

# ==========================================
# KONFIGURASI PATH (SESUAIKAN DENGAN KOMPUTER ANDA)
# ==========================================
BASE_DIR = 'C:/zafaa/kuliah/SEMESTER7/PRATA/BukuTATekkomLatex'

# Direktori output gambar CM
OUTPUT_DIR = os.path.join(BASE_DIR, 'gambar/bab-4/confusion_matrix_pengujian')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path output file LaTeX
LATEX_OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
LATEX_FILE_PATH = os.path.join(LATEX_OUTPUT_DIR, 'akurasi_pengujian_all.tex')
os.makedirs(LATEX_OUTPUT_DIR, exist_ok=True)

# PATH KE FILE CSV DATA PENGUJIAN DI RASPI/LAPTOP
# Pastikan file ini sudah berisi kolom 'Target_Gesture' dan 'Last_Command'
CSV_PATH = 'C:/zafaa/kuliah/SEMESTER7/PRATA/code_gesture/LSTM_HandGesture/data/data_pengujian.csv'
# -----------------------------------------------------


# ==========================================
# FUNGSI-FUNGSI HELPER VISUALISASI & LATEX
# ==========================================
def generate_manual_cm(data_matrix, title, filename_suffix):
    """Fungsi untuk menghasilkan dan menyimpan gambar Confusion Matrix."""
    expected_size = len(CLASS_LABELS)
    # Validasi bentuk matriks
    if data_matrix.shape != (expected_size, expected_size):
        print(f"⚠️ WARNING: Ukuran matriks untuk {title} adalah {data_matrix.shape}, seharusnya {expected_size}x{expected_size}. Gambar mungkin tidak akurat.")
        # Kita tetap lanjut agar tidak crash, tapi hasilnya mungkin aneh

    plt.figure(figsize=(10, 8))
    sns.heatmap(data_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                annot_kws={"size": 12})

    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('Label Aktual (Target Seharusnya)', fontsize=12) # Sumbu Y
    plt.xlabel('Label Prediksi Sistem (Realisasi)', fontsize=12) # Sumbu X
    plt.tight_layout()

    filepath = os.path.join(OUTPUT_DIR, f'cm_{filename_suffix}.png')
    plt.savefig(filepath, dpi=300)
    print(f"✅ Gambar berhasil disimpan: {filepath}")
    plt.close()

def hitung_akurasi_dan_generate_latex(data_matrix, scenario_prefix):
    """Menghitung akurasi per kelas dari CM dan membuat command LaTeX."""
    latex_commands = []
    
    # Pastikan matriks adalah numpy array 8x8
    if data_matrix.shape != (8, 8):
         data_matrix = np.zeros((8,8), dtype=int)

    true_positives = np.diag(data_matrix)
    total_per_class = np.sum(data_matrix, axis=1)

    accuracies = []
    for i, label in enumerate(CLASS_LABELS):
        if total_per_class[i] > 0:
            # Rumus: (Benar / Total Sampel Kelas Itu) * 100
            acc = (true_positives[i] / total_per_class[i]) * 100
        else:
            acc = 0.0
        accuracies.append(acc)
        cmd_name = f"Acc{scenario_prefix}{LABEL_TO_CMD[label]}"
        latex_commands.append(f"\\newcommand{{\\{cmd_name}}}{{{acc:.1f}}}")

    # Hitung rata-rata total hanya dari kelas yang memiliki data pengujian
    kelas_yg_ada_data = [acc for i, acc in enumerate(accuracies) if total_per_class[i] > 0]
    if kelas_yg_ada_data:
        avg_acc = np.mean(kelas_yg_ada_data)
    else:
        avg_acc = 0.0

    avg_cmd_name = f"Acc{scenario_prefix}Avg"
    latex_commands.append(f"\\newcommand{{\\{avg_cmd_name}}}{{{avg_acc:.1f}}}")

    return "\n".join(latex_commands)

# ==============================================================================
# --- FUNGSI BARU UTAMA: MEMBANGUN CM RIIL DARI CSV ---
# ==============================================================================
def build_real_cm_from_csv(df, filter_col, filter_val):
    """
    Membangun Confusion Matrix 8x8 Riil dari data CSV menggunakan pd.crosstab.
    - Sumbu Y (Aktual) diambil dari kolom 'Target_Gesture'
    - Sumbu X (Prediksi) diambil dari kolom 'Last_Command'
    """
    print(f"   [Processing] Memfilter data: {filter_col} == {filter_val}...")
    
    # 1. Filter data berdasarkan kriteria (misal: Resolution == '480p')
    filtered_df = df[df[filter_col] == filter_val]

    if filtered_df.empty:
        print(f"   [Info] Data kosong untuk filter ini. Mengembalikan matriks 0.")
        return np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=int)

    # 2. Validasi: Pastikan kolom target dan command ada
    if 'Target_Gesture' not in filtered_df.columns or 'Last_Command' not in filtered_df.columns:
        print("❌ ERROR: Kolom 'Target_Gesture' atau 'Last_Command' tidak ditemukan di CSV.")
        return np.zeros((8,8), dtype=int)

    # 3. GUNAKAN PANDAS CROSSTAB (Ini inti otomatisasinya)
    # Ini otomatis menghitung frekuensi silang antara Aktual vs Prediksi
    cm_df = pd.crosstab(
        index=filtered_df['Target_Gesture'],   # Baris = Aktual
        columns=filtered_df['Last_Command'],   # Kolom = Prediksi
        dropna=False # Jangan buang kategori yang kosong dulu
    )
    
    # 4. REINDEXING (PENTING!)
    # Memastikan matriks selalu berukuran 8x8 sesuai urutan CLASS_LABELS,
    # meskipun ada gestur yang tidak pernah muncul di data.
    # Nilai yang kosong diisi dengan 0.
    cm_df_reindexed = cm_df.reindex(index=CLASS_LABELS, columns=CLASS_LABELS, fill_value=0)

    print(f"   [Success] CM berhasil dibangun dari {len(filtered_df)} sampel data.")
    # Ubah dari DataFrame pandas menjadi numpy array agar kompatibel dengan fungsi visualisasi
    return cm_df_reindexed.to_numpy()
# ==============================================================================


# ==========================================
# MAIN PROGRAM
# ==========================================
if __name__ == "__main__":

    # ===================================================
    # BAGIAN 1: LOAD CSV DATA
    # ===================================================
    print(f"--- Membaca data dari CSV: {CSV_PATH} ---")
    df_csv = pd.DataFrame() # Inisialisasi DF kosong
    try:
        df_csv = pd.read_csv(CSV_PATH)
        print(f"✅ Berhasil membaca {len(df_csv)} baris data.")
        
        # Bersihkan spasi ekstra jika ada di data CSV agar matching string sempurna
        if not df_csv.empty:
            if 'Target_Gesture' in df_csv.columns:
                df_csv['Target_Gesture'] = df_csv['Target_Gesture'].astype(str).str.strip()
            if 'Last_Command' in df_csv.columns:
                df_csv['Last_Command'] = df_csv['Last_Command'].astype(str).str.strip()
                
    except FileNotFoundError:
        print(f"⚠️ WARNING: File {CSV_PATH} tidak ditemukan.")
        print("Semua matriks otomatis akan kosong (nol).")
    except Exception as e:
        print(f"❌ ERROR saat membaca CSV: {e}")
        exit()


    # ===================================================
    # BAGIAN 2: DEFINISI DATA MATRIKS
    # ===================================================

    # ---------------------------------------------------
    # A. SKENARIO OTOMATIS DARI CSV (VARIASI JARAK)
    # ---------------------------------------------------
    print("\n--- Memproses Data Otomatis dari CSV (Variasi Jarak) ---")
    # Pastikan nilai filter ('30cm', dll) SAMA PERSIS dengan yang tertulis di CSV Anda
    cm_data_jarak_30 = build_real_cm_from_csv(df_csv, 'Distance', '30cm')
    generate_manual_cm(cm_data_jarak_30, "Confusion Matrix - Jarak 30 cm", "jarak_30cm")

    cm_data_jarak_50 = build_real_cm_from_csv(df_csv, 'Distance', '50cm')
    generate_manual_cm(cm_data_jarak_50, "Confusion Matrix - Jarak 50 cm", "jarak_50cm")
    
    cm_data_jarak_70 = build_real_cm_from_csv(df_csv, 'Distance', '70cm')
    generate_manual_cm(cm_data_jarak_70, "Confusion Matrix - Jarak 70 cm", "jarak_70cm")


    # ---------------------------------------------------
    # B. SKENARIO OTOMATIS DARI CSV (VARIASI RESOLUSI)
    # ---------------------------------------------------
    print("\n--- Memproses Data Otomatis dari CSV (Variasi Resolusi) ---")
    # 480p
    cm_data_resolusi_480p = build_real_cm_from_csv(df_csv, 'Resolution', '480p')
    generate_manual_cm(cm_data_resolusi_480p, "Confusion Matrix - Resolusi 480p", "resolusi_480p")

    # 720p
    cm_data_resolusi_720p = build_real_cm_from_csv(df_csv, 'Resolution', '720p')
    generate_manual_cm(cm_data_resolusi_720p, "Confusion Matrix - Resolusi 720p", "resolusi_720p")


    # ---------------------------------------------------
    # C. SKENARIO MANUAL DUMMY (VARIASI PENCAHAYAAN)
    # Karena datanya belum ada, kita pakai data dummy dulu.
    # Nanti kalau datanya sudah ada di CSV (misal ada kolom 'Lighting'),
    # tinggal ubah jadi pakai build_real_cm_from_csv juga.
    # ---------------------------------------------------
    print("\n--- Memproses Data Manual DUMMY (Variasi Pencahayaan) ---")
    # Cahaya Redup (DUMMY)
    cm_data_cahaya_redup = np.array([[15, 2, 1, 0, 0, 0, 0, 0], [3, 14, 0, 1, 0, 0, 0, 0], [1, 0, 16, 3, 0, 0, 0, 0], [0, 0, 2, 18, 0, 0, 0, 0], [0, 0, 0, 0, 12, 5, 1, 0], [0, 0, 0, 0, 4, 16, 0, 0], [0, 0, 0, 0, 1, 0, 14, 3], [0, 1, 0, 0, 0, 0, 2, 17]])
    generate_manual_cm(cm_data_cahaya_redup, "Confusion Matrix - Cahaya Redup (DUMMY)", "cahaya_redup")
    # Cahaya Sedang (DUMMY)
    cm_data_cahaya_sedang = np.array([[20, 0, 0, 0, 0, 0, 0, 0], [0, 20, 0, 0, 0, 0, 0, 0], [0, 0, 20, 0, 0, 0, 0, 0], [0, 0, 0, 20, 0, 0, 0, 0], [0, 0, 0, 0, 20, 0, 0, 0], [0, 0, 0, 0, 0, 20, 0, 0], [0, 0, 0, 0, 0, 0, 20, 0], [0, 0, 0, 0, 0, 0, 0, 20]])
    generate_manual_cm(cm_data_cahaya_sedang, "Confusion Matrix - Cahaya Sedang (DUMMY)", "cahaya_sedang")
    # Cahaya Terang (DUMMY)
    cm_data_cahaya_terang = np.array([[19, 1, 0, 0, 0, 0, 0, 0], [0, 20, 0, 0, 0, 0, 0, 0], [0, 0, 18, 2, 0, 0, 0, 0], [0, 0, 0, 20, 0, 0, 0, 0], [0, 0, 0, 0, 19, 1, 0, 0], [0, 0, 0, 0, 1, 19, 0, 0], [0, 0, 0, 0, 0, 0, 20, 0], [0, 0, 0, 0, 0, 0, 0, 20]])
    generate_manual_cm(cm_data_cahaya_terang, "Confusion Matrix - Cahaya Terang (DUMMY)", "cahaya_terang")


    # ===================================================
    # BAGIAN 3: MENGHITUNG TOTAL DAN GENERATE LATEX
    # ===================================================
    print("\n--- Menghitung Total Confusion Matrix Gabungan ---")

    # Kumpulkan semua matriks
    semua_matriks_skenario = [
        cm_data_jarak_30, cm_data_jarak_50, cm_data_jarak_70,
        cm_data_cahaya_redup, cm_data_cahaya_sedang, cm_data_cahaya_terang,
        cm_data_resolusi_480p, cm_data_resolusi_720p
    ]

    # Jumlahkan semua matriks (pastikan semuanya numpy array)
    cm_data_total_gabungan = np.sum([m for m in semua_matriks_skenario if isinstance(m, np.ndarray)], axis=0)

    # Generate gambar total
    generate_manual_cm(
        data_matrix=cm_data_total_gabungan,
        title="Confusion Matrix Total Gabungan Pengujian",
        filename_suffix="total_gabungan"
    )

    print("\n--- Semua gambar Confusion Matrix selesai dibuat ---")

    # ===================================================
    # BAGIAN 4: GENERATE FILE LATEX
    # ===================================================

    # Daftarkan semua matriks dan prefixnya
    data_skenario = {
        "JarakDekat": cm_data_jarak_30,
        "JarakIdeal": cm_data_jarak_50,
        "JarakJauh":  cm_data_jarak_70,
        "ResSedang": cm_data_resolusi_480p,
        "ResTinggi": cm_data_resolusi_720p,
        "CahayaRedup":  cm_data_cahaya_redup, # Masih dummy
        "CahayaSedang": cm_data_cahaya_sedang, # Masih dummy
        "CahayaTerang": cm_data_cahaya_terang, # Masih dummy
        "TotalGabungan": cm_data_total_gabungan
    }

    print("--- Memulai Proses Generasi File LaTeX ---")
    full_latex_content = "% File DIGENERATE OTOMATIS OLEH PYTHON DARI DATA CSV RIIL & DUMMY.\n"
    full_latex_content += "% JANGAN DIEDIT MANUAL.\n"
    full_latex_content += "% Waktu generate: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n"

    for prefix, matrix in data_skenario.items():
        print(f"Memproses data LaTeX untuk: {prefix}...")
        latex_block = f"% --- Data Akurasi untuk Skenario: {prefix} ---\n"
        # Pastikan matriks valid sebelum dihitung
        if isinstance(matrix, np.ndarray):
            latex_block += hitung_akurasi_dan_generate_latex(matrix, prefix)
        else:
            latex_block += f"% ERROR: Matriks untuk {prefix} tidak valid/kosong.\n"
        latex_block += "\n\n"
        full_latex_content += latex_block

    try:
        with open(LATEX_FILE_PATH, "w") as f:
            f.write(full_latex_content)
        print(f"\n✅ SUKSES! File data LaTeX berhasil diperbarui di:\n{LATEX_FILE_PATH}")
        print("Periksa file tersebut untuk melihat nilai akurasi yang baru.")
    except Exception as e:
        print(f"\n❌ ERROR saat menulis file LaTeX: {e}")