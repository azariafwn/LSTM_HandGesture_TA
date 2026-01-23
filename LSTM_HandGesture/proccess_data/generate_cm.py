import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd # Import Pandas wajib ada

# ==========================================
# KONFIGURASI LABEL KELAS
# ==========================================
# Urutan HARUS sama persis dengan yang ada di CSV
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
# KONFIGURASI PATH
# ==========================================
BASE_DIR = 'C:/zafaa/kuliah/SEMESTER7/PRATA/BukuTATekkomLatex'

# Direktori output gambar CM
OUTPUT_DIR = os.path.join(BASE_DIR, 'gambar/bab-4/confusion_matrix_pengujian')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path output file LaTeX
LATEX_OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
LATEX_FILE_PATH = os.path.join(LATEX_OUTPUT_DIR, 'akurasi_pengujian_all.tex')
os.makedirs(LATEX_OUTPUT_DIR, exist_ok=True)

# PATH KE FILE CSV DATA PENGUJIAN
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

# --- FUNGSI INI YANG DIUPDATE ---
def hitung_akurasi_dan_generate_latex(data_matrix, scenario_prefix):
    """
    Menghitung akurasi (persen dengan 1 desimal) dan jumlah sampel (integer tanpa koma).
    """
    latex_commands = []
    
    # Pastikan matriks adalah numpy array 8x8
    if data_matrix.shape != (8, 8):
         data_matrix = np.zeros((8,8), dtype=int)

    true_positives = np.diag(data_matrix)
    # total_per_class adalah penjumlahan baris (Jumlah sampel aktual tiap kelas)
    total_per_class = np.sum(data_matrix, axis=1)
    
    # Hitung total sampel keseluruhan dalam skenario ini
    total_scenario_samples = np.sum(data_matrix)

    accuracies = []
    for i, label in enumerate(CLASS_LABELS):
        # 1. Hitung Akurasi Per Kelas
        if total_per_class[i] > 0:
            acc = (true_positives[i] / total_per_class[i]) * 100
        else:
            acc = 0.0
        accuracies.append(acc)
        acc_cmd_name = f"Acc{scenario_prefix}{LABEL_TO_CMD[label]}"
        latex_commands.append(f"\\newcommand{{\\{acc_cmd_name}}}{{{acc:.1f}}}")

        # 2. Hitung Jumlah Sampel Per Kelas
        count_cmd_name = f"Count{scenario_prefix}{LABEL_TO_CMD[label]}"
        latex_commands.append(f"\\newcommand{{\\{count_cmd_name}}}{{{int(total_per_class[i])}}}")

    # 3. Hitung Rata-rata Akurasi Skenario
    kelas_yg_ada_data = [acc for i, acc in enumerate(accuracies) if total_per_class[i] > 0]
    if kelas_yg_ada_data:
        avg_acc = np.mean(kelas_yg_ada_data)
    else:
        avg_acc = 0.0
    avg_cmd_name = f"Acc{scenario_prefix}Avg"
    latex_commands.append(f"\\newcommand{{\\{avg_cmd_name}}}{{{avg_acc:.1f}}}")

    # 4. Hitung Total Sampel Skenario
    total_cmd_name = f"Count{scenario_prefix}Total"
    latex_commands.append(f"\\newcommand{{\\{total_cmd_name}}}{{{int(total_scenario_samples)}}}")

    ### --- Rata-rata Count ---
    num_active_classes = np.sum(total_per_class > 0)
    
    if num_active_classes > 0:
        avg_count_per_class = total_scenario_samples / num_active_classes
    else:
        avg_count_per_class = 0
        
    avg_count_cmd_name = f"Count{scenario_prefix}Avg"
    latex_commands.append(f"\\newcommand{{\\{avg_count_cmd_name}}}{{{int(avg_count_per_class)}}}")
    ### ------------------------------------

    return "\n".join(latex_commands)
# --------------------------------

# ==============================================================================
# --- FUNGSI MEMBANGUN CM RIIL DARI CSV ---
# ==============================================================================
def build_real_cm_from_csv(df, filter_col, filter_val):
    print(f"   [Processing] Memfilter data: {filter_col} == {filter_val}...")
    filtered_df = df[df[filter_col] == filter_val]

    if filtered_df.empty:
        print(f"   [Info] Data kosong untuk filter ini. Mengembalikan matriks 0.")
        return np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype=int)

    if 'Target_Gesture' not in filtered_df.columns or 'Last_Command' not in filtered_df.columns:
        print("❌ ERROR: Kolom 'Target_Gesture' atau 'Last_Command' tidak ditemukan di CSV.")
        return np.zeros((8,8), dtype=int)

    cm_df = pd.crosstab(
        index=filtered_df['Target_Gesture'],
        columns=filtered_df['Last_Command'],
        dropna=False
    )
    
    cm_df_reindexed = cm_df.reindex(index=CLASS_LABELS, columns=CLASS_LABELS, fill_value=0)
    print(f"   [Success] CM berhasil dibangun dari {len(filtered_df)} sampel data.")
    return cm_df_reindexed.to_numpy()


# ==========================================
# MAIN PROGRAM
# ==========================================
if __name__ == "__main__":

    # ===================================================
    # BAGIAN 1: LOAD CSV DATA
    # ===================================================
    print(f"--- Membaca data dari CSV: {CSV_PATH} ---")
    df_csv = pd.DataFrame()
    try:
        df_csv = pd.read_csv(CSV_PATH)
        print(f"✅ Berhasil membaca {len(df_csv)} baris data.")
        
        if not df_csv.empty:
            if 'Target_Gesture' in df_csv.columns:
                df_csv['Target_Gesture'] = df_csv['Target_Gesture'].astype(str).str.strip()
            if 'Last_Command' in df_csv.columns:
                df_csv['Last_Command'] = df_csv['Last_Command'].astype(str).str.strip()
                
    except FileNotFoundError:
        print(f"⚠️ WARNING: File {CSV_PATH} tidak ditemukan.")
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
    cm_data_resolusi_480p = build_real_cm_from_csv(df_csv, 'Resolution', '480p')
    generate_manual_cm(cm_data_resolusi_480p, "Confusion Matrix - Resolusi 480p", "resolusi_480p")

    cm_data_resolusi_720p = build_real_cm_from_csv(df_csv, 'Resolution', '720p')
    generate_manual_cm(cm_data_resolusi_720p, "Confusion Matrix - Resolusi 720p", "resolusi_720p")


    # ---------------------------------------------------
    # C. SKENARIO MANUAL DUMMY (VARIASI PENCAHAYAAN)
    # ---------------------------------------------------
    print("\n--- Memproses Data Manual DUMMY (Variasi Pencahayaan) ---")
    cm_data_cahaya_redup = np.zeros((8,8), dtype=int)
    generate_manual_cm(cm_data_cahaya_redup, "Confusion Matrix - Cahaya Redup (DUMMY)", "cahaya_redup")
    
    cm_data_cahaya_sedang = np.zeros((8,8), dtype=int)
    generate_manual_cm(cm_data_cahaya_sedang, "Confusion Matrix - Cahaya Sedang (DUMMY)", "cahaya_sedang")
    
    cm_data_cahaya_terang = np.zeros((8,8), dtype=int)
    generate_manual_cm(cm_data_cahaya_terang, "Confusion Matrix - Cahaya Terang (DUMMY)", "cahaya_terang")


    # ===================================================
    # BAGIAN 3: MENGHITUNG TOTAL DAN GENERATE LATEX
    # ===================================================
    print("\n--- Menghitung Total Confusion Matrix Gabungan ---")

    semua_matriks_skenario = [
        cm_data_jarak_30, cm_data_jarak_50, cm_data_jarak_70,
        cm_data_cahaya_redup, cm_data_cahaya_sedang, cm_data_cahaya_terang,
        cm_data_resolusi_480p, cm_data_resolusi_720p
    ]

    cm_data_total_gabungan = np.sum([m for m in semua_matriks_skenario if isinstance(m, np.ndarray)], axis=0)

    generate_manual_cm(
        data_matrix=cm_data_total_gabungan,
        title="Confusion Matrix Total Gabungan Pengujian",
        filename_suffix="total_gabungan"
    )

    print("\n--- Semua gambar Confusion Matrix selesai dibuat ---")

    # ===================================================
    # BAGIAN 4: GENERATE FILE LATEX
    # ===================================================
    data_skenario = {
        "JarakDekat": cm_data_jarak_30,
        "JarakIdeal": cm_data_jarak_50,
        "JarakJauh":  cm_data_jarak_70,
        "ResSedang": cm_data_resolusi_480p,
        "ResTinggi": cm_data_resolusi_720p,
        "CahayaRedup":  cm_data_cahaya_redup,
        "CahayaSedang": cm_data_cahaya_sedang,
        "CahayaTerang": cm_data_cahaya_terang,
        "TotalGabungan": cm_data_total_gabungan
    }

    print("--- Memulai Proses Generasi File LaTeX ---")
    full_latex_content = "% File DIGENERATE OTOMATIS OLEH PYTHON.\n"
    full_latex_content += "% Berisi Akurasi (%), Jumlah Sampel (Count), dan Rata-rata Sampel (CountAvg).\n"
    full_latex_content += "% Waktu generate: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n"

    for prefix, matrix in data_skenario.items():
        print(f"Memproses data LaTeX untuk: {prefix}...")
        latex_block = f"% --- Data untuk Skenario: {prefix} ---\n"
        if isinstance(matrix, np.ndarray):
            latex_block += hitung_akurasi_dan_generate_latex(matrix, prefix)
        else:
            latex_block += f"% ERROR: Matriks untuk {prefix} tidak valid.\n"
        latex_block += "\n\n"
        full_latex_content += latex_block

    try:
        with open(LATEX_FILE_PATH, "w") as f:
            f.write(full_latex_content)
        print(f"\n✅ SUKSES! File data LaTeX berhasil diperbarui di:\n{LATEX_FILE_PATH}")
    except Exception as e:
        print(f"\n❌ ERROR saat menulis file LaTeX: {e}")