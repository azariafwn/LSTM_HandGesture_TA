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

def hitung_akurasi_dan_generate_latex(data_matrix, scenario_prefix, calc_extra_metrics=False):
    """
    Menghitung akurasi dan jumlah sampel.
    Jika calc_extra_metrics=True, hitung juga Precision, Recall, dan F1-Score.
    UPDATE: Format per kelas jadi desimal 2 angka (0.98), rata-rata jadi persen 2 angka (98.55).
    """
    latex_commands = []
    
    if data_matrix.shape != (8, 8):
         data_matrix = np.zeros((8,8), dtype=int)

    true_positives = np.diag(data_matrix)
    total_actual_per_class = np.sum(data_matrix, axis=1)
    total_predicted_per_class = np.sum(data_matrix, axis=0)
    total_scenario_samples = np.sum(data_matrix)

    accuracies = [] # Menyimpan persen
    precisions_dec = [] # Menyimpan desimal
    recalls_dec = [] # Menyimpan desimal
    f1_scores_dec = [] # Menyimpan desimal

    for i, label in enumerate(CLASS_LABELS):
        cmd_suffix = LABEL_TO_CMD[label]

        # --- 1. Hitung Akurasi Per Kelas (Tetap Persen, 1 Desimal) ---
        if total_actual_per_class[i] > 0:
            acc = (true_positives[i] / total_actual_per_class[i]) * 100
        else:
            acc = 0.0
        accuracies.append(acc)
        latex_commands.append(f"\\newcommand{{\\Acc{scenario_prefix}{cmd_suffix}}}{{{acc:.1f}}}")

        # --- 2. Hitung Jumlah Sampel Per Kelas ---
        latex_commands.append(f"\\newcommand{{\\Count{scenario_prefix}{cmd_suffix}}}{{{int(total_actual_per_class[i])}}}")

        # ============================================================
        # BAGIAN BARU: Hitung Precision, Recall, F1 (Jika diminta)
        # UPDATE: FORMAT DESIMAL 2 ANGKA BELAKANG KOMA (misal 1.00, 0.98)
        # ============================================================
        if calc_extra_metrics:
            tp = true_positives[i]
            actual = total_actual_per_class[i]
            predicted = total_predicted_per_class[i]

            # --- Precision (Decimal) ---
            # Hapus pengali 100 agar jadi desimal (0.0 - 1.0)
            precision_dec = (tp / predicted) if predicted > 0 else 0.0
            precisions_dec.append(precision_dec)
            # Format .2f
            latex_commands.append(f"\\newcommand{{\\Prec{scenario_prefix}{cmd_suffix}}}{{{precision_dec:.2f}}}")

            # --- Recall (Decimal) ---
            # Hapus pengali 100
            recall_dec = (tp / actual) if actual > 0 else 0.0
            recalls_dec.append(recall_dec)
            # Format .2f
            latex_commands.append(f"\\newcommand{{\\Recall{scenario_prefix}{cmd_suffix}}}{{{recall_dec:.2f}}}")

            # --- F1-Score (Decimal) ---
            # Rumus menggunakan nilai desimal
            if (precision_dec + recall_dec) > 0:
                f1_score_dec = 2 * (precision_dec * recall_dec) / (precision_dec + recall_dec)
            else:
                f1_score_dec = 0.0
            f1_scores_dec.append(f1_score_dec)
            # Format .2f
            latex_commands.append(f"\\newcommand{{\\FOne{scenario_prefix}{cmd_suffix}}}{{{f1_score_dec:.2f}}}")
        # ============================================================


    # ============================================================
    # BAGIAN RATA-RATA (MACRO AVERAGE)
    # ============================================================
    def calculate_macro_average(metrics_list, actual_counts):
        valid_metrics = [m for i, m in enumerate(metrics_list) if actual_counts[i] > 0]
        return np.mean(valid_metrics) if valid_metrics else 0.0

    # --- Rata-rata Akurasi (Tetap Persen, 1 Desimal) ---
    avg_acc = calculate_macro_average(accuracies, total_actual_per_class)
    latex_commands.append(f"\\newcommand{{\\Acc{scenario_prefix}Avg}}{{{avg_acc:.1f}}}")

    # --- Rata-rata Metrik Tambahan (Jika diminta) ---
    # UPDATE: FORMAT PERSEN 2 ANGKA BELAKANG KOMA (misal 98.55)
    if calc_extra_metrics:
        # Rata-rata Precision (Input desimal -> Output dikali 100 jadi persen)
        avg_prec_dec = calculate_macro_average(precisions_dec, total_actual_per_class)
        latex_commands.append(f"\\newcommand{{\\Prec{scenario_prefix}Avg}}{{{(avg_prec_dec * 100):.2f}}}")

        # Rata-rata Recall
        avg_recall_dec = calculate_macro_average(recalls_dec, total_actual_per_class)
        latex_commands.append(f"\\newcommand{{\\Recall{scenario_prefix}Avg}}{{{(avg_recall_dec * 100):.2f}}}")

        # Rata-rata F1-Score
        avg_f1_dec = calculate_macro_average(f1_scores_dec, total_actual_per_class)
        latex_commands.append(f"\\newcommand{{\\FOne{scenario_prefix}Avg}}{{{(avg_f1_dec * 100):.2f}}}")
    # ============================================================

    # Hitung Total dan Rata-rata Count
    latex_commands.append(f"\\newcommand{{\\Count{scenario_prefix}Total}}{{{int(total_scenario_samples)}}}")
    num_active_classes = np.sum(total_actual_per_class > 0)
    avg_count_per_class = total_scenario_samples / num_active_classes if num_active_classes > 0 else 0
    latex_commands.append(f"\\newcommand{{\\Count{scenario_prefix}Avg}}{{{int(avg_count_per_class)}}}")

    return "\n".join(latex_commands)

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
            if 'Light_Intensity_Lux' in df_csv.columns:
                df_csv['Light_Intensity_Lux'] = pd.to_numeric(df_csv['Light_Intensity_Lux'], errors='coerce')
    except FileNotFoundError:
        print(f"⚠️ WARNING: File {CSV_PATH} tidak ditemukan.")
    except Exception as e:
        print(f"❌ ERROR saat membaca CSV: {e}")
        exit()

    # ===================================================
    # BAGIAN 2: DEFINISI DATA MATRIKS
    # ===================================================
    # A. Variasi Jarak
    print("\n--- Memproses Data Otomatis dari CSV (Variasi Jarak) ---")
    cm_data_jarak_30 = build_real_cm_from_csv(df_csv, 'Distance', '30cm')
    generate_manual_cm(cm_data_jarak_30, "Confusion Matrix - Jarak 30 cm", "jarak_30cm")
    cm_data_jarak_50 = build_real_cm_from_csv(df_csv, 'Distance', '50cm')
    generate_manual_cm(cm_data_jarak_50, "Confusion Matrix - Jarak 50 cm", "jarak_50cm")
    cm_data_jarak_70 = build_real_cm_from_csv(df_csv, 'Distance', '70cm')
    generate_manual_cm(cm_data_jarak_70, "Confusion Matrix - Jarak 70 cm", "jarak_70cm")

    # B. Variasi Resolusi
    print("\n--- Memproses Data Otomatis dari CSV (Variasi Resolusi) ---")
    cm_data_resolusi_480p = build_real_cm_from_csv(df_csv, 'Resolution', '480p')
    generate_manual_cm(cm_data_resolusi_480p, "Confusion Matrix - Resolusi 480p", "resolusi_480p")
    cm_data_resolusi_720p = build_real_cm_from_csv(df_csv, 'Resolution', '720p')
    generate_manual_cm(cm_data_resolusi_720p, "Confusion Matrix - Resolusi 720p", "resolusi_720p")

    # C. Variasi Pencahayaan
    print("\n--- Memproses Data Otomatis dari CSV (Variasi Pencahayaan) ---")
    cm_data_cahaya_redup = build_real_cm_from_csv(df_csv, 'Light_Intensity_Lux', 30)
    generate_manual_cm(cm_data_cahaya_redup, "Confusion Matrix - Cahaya Redup (±30 Lux)", "cahaya_redup")
    cm_data_cahaya_sedang = build_real_cm_from_csv(df_csv, 'Light_Intensity_Lux', 150)
    generate_manual_cm(cm_data_cahaya_sedang, "Confusion Matrix - Cahaya Sedang (±150 Lux)", "cahaya_sedang")
    cm_data_cahaya_terang = build_real_cm_from_csv(df_csv, 'Light_Intensity_Lux', 400)
    generate_manual_cm(cm_data_cahaya_terang, "Confusion Matrix - Cahaya Terang (±400 Lux)", "cahaya_terang")

    # ===================================================
    # BAGIAN 3: MENGHITUNG TOTAL DAN GENERATE LATEX
    # ===================================================
    print("\n--- Menghitung Total Confusion Matrix Gabungan ---")
    semua_matriks_skenario = [
        cm_data_jarak_30, cm_data_jarak_50, cm_data_jarak_70,
        cm_data_resolusi_480p, cm_data_resolusi_720p,
        cm_data_cahaya_redup, cm_data_cahaya_sedang, cm_data_cahaya_terang
    ]
    cm_data_total_gabungan = np.sum([m for m in semua_matriks_skenario if isinstance(m, np.ndarray)], axis=0)
    generate_manual_cm(cm_data_total_gabungan, "Confusion Matrix Total Gabungan Pengujian", "total_gabungan")
    print("\n--- Semua gambar Confusion Matrix selesai dibuat ---")

    # ===================================================
    # BAGIAN 4: GENERATE FILE LATEX
    # ===================================================
    data_skenario = {
        "JarakDekat": cm_data_jarak_30, "JarakIdeal": cm_data_jarak_50, "JarakJauh": cm_data_jarak_70,
        "ResSedang": cm_data_resolusi_480p, "ResTinggi": cm_data_resolusi_720p,
        "CahayaRedup": cm_data_cahaya_redup, "CahayaSedang": cm_data_cahaya_sedang, "CahayaTerang": cm_data_cahaya_terang,
        "TotalGabungan": cm_data_total_gabungan
    }

    print("--- Memulai Proses Generasi File LaTeX ---")
    full_latex_content = "% File DIGENERATE OTOMATIS OLEH PYTHON.\n"
    full_latex_content += "% Waktu generate: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n"

    for prefix, matrix in data_skenario.items():
        print(f"Memproses data LaTeX untuk: {prefix}...")
        latex_block = f"% --- Data untuk Skenario: {prefix} ---\n"
        if isinstance(matrix, np.ndarray):
            need_extra_metrics = (prefix == "TotalGabungan")
            latex_block += hitung_akurasi_dan_generate_latex(matrix, prefix, calc_extra_metrics=need_extra_metrics)
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