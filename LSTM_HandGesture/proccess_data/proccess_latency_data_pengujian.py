import pandas as pd
import os
import numpy as np
import re

# === KONFIGURASI ===
# PASTIIN PATH INI BENER DI KOMPUTER KAMU
CSV_FILE_PATH = 'C:/zafaa/kuliah/SEMESTER7/PRATA/code_gesture/LSTM_HandGesture/data/data_pengujian.csv'
LATEX_PROJECT_DIR = 'C:/zafaa/kuliah/SEMESTER7/PRATA/BukuTATekkomLatex'
TEX_DATA_DIR = os.path.join(LATEX_PROJECT_DIR, 'data/')
TEX_OUTPUT_PATH = os.path.join(TEX_DATA_DIR, 'latency_pengujian_stats.tex')

# Definisi metrik yang akan dihitung rata-ratanya per skenario
METRICS_TO_ANALYZE = {
    'FPS': 'FPS',
    'Edge_Latency_ms': 'Edge',
    'WiFi_Latency_ms': 'WiFi',
    'Total_Latency_ms': 'Total'
}

# Definisi kolom skenario di CSV dan prefix untuk command LaTeX-nya
SCENARIO_COLUMNS = {
    'Distance': 'Dist',    
    'Resolution': 'Res',   
    # 'Lighting': 'Light'  # Uncomment nanti jika data cahaya sudah ada
}


def fmt(val):
    """Helper untuk format angka ke string 2 desimal."""
    if pd.isna(val) or np.isnan(val):
            return "0.00"
    return f"{val:.2f}"

# --- PERBAIKAN DI SINI ---
def clean_label_for_latex(label):
    """
    Membersihkan string agar aman jadi nama command LaTeX.
    ATURAN PENTING: Command LaTeX HANYA boleh huruf (A-Z, a-z), TANPA ANGKA.
    """
    label_str = str(label)
    
    # Mapping manual untuk mengubah angka menjadi teks
    # Agar nama command tetap unik dan terbaca
    replacements = {
        "30cm": "ThirtyCm",
        "50cm": "FiftyCm",
        "70cm": "SeventyCm",
        "480p": "FourEightyP",
        "720p": "SevenTwentyP",
        # Tambahkan mapping lain jika ada nilai baru yang mengandung angka
        "Redup": "Redup",   # Sudah aman
        "Sedang": "Sedang", # Sudah aman
        "Terang": "Terang"  # Sudah aman
    }

    if label_str in replacements:
        return replacements[label_str]
    else:
        # Fallback: Hapus semua karakter selain huruf
        # Ini untuk jaga-jaga jika ada data tak terduga
        cleaned = re.sub(r'[^a-zA-Z]', '', label_str)
        # Pastikan huruf pertama kapital (CamelCase)
        return cleaned.capitalize() if cleaned else "Unknown"
# -------------------------

def generate_scenario_stats_latex(df):
    """Fungsi baru untuk menghasilkan statistik spesifik per skenario."""
    latex_output = "\n% ==================================================\n"
    latex_output += "% STATISTIK SPESIFIK PER SKENARIO (RATA-RATA)\n"
    latex_output += "% Digunakan untuk tabel perbandingan di Bab Pengujian\n"
    latex_output += "% ==================================================\n"

    for col_name, prefix in SCENARIO_COLUMNS.items():
        if col_name not in df.columns:
            print(f"⚠️ Peringatan: Kolom '{col_name}' tidak ditemukan di CSV. Melewati analisis skenario ini.")
            latex_output += f"\n% --- Skenario: {col_name} (DATA TIDAK DITEMUKAN) ---\n"
            continue
            
        print(f"   [Processing] Menganalisis variasi berdasarkan: {col_name}...")
        latex_output += f"\n% --- Skenario Variasi: {col_name} ---\n"
        
        # Kelompokkan data
        grouped = df.groupby(col_name)
        
        for group_val, group_df in grouped:
            # Gunakan fungsi pembersih yang baru
            clean_val = clean_label_for_latex(group_val) 
            latex_output += f"% Group: {group_val} (Sampel: {len(group_df)})\n"
            
            # Hitung rata-rata untuk setiap metrik
            for metric_col, metric_prefix in METRICS_TO_ANALYZE.items():
                mean_val = group_df[metric_col].mean()
                
                # Buat nama command: \PrefixMetricMeanPrefixSkenarioValueSkenarioBersih
                cmd_name = f"{metric_prefix}Mean{prefix}{clean_val}"
                latex_output += f"\\newcommand{{\\{cmd_name}}}{{{fmt(mean_val)}}}\n"
            
            latex_output += "\n"
            
    return latex_output

def process_data():
    print(f"--- Memulai pemrosesan data dari {CSV_FILE_PATH} ---")

    # 1. Baca data dari CSV
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        total_samples = len(df)
        print(f"✅ Berhasil membaca {total_samples} baris data.")
        
        # Pastikan kolom numerik dibaca sebagai angka
        for col in ['FPS', 'Edge_Latency_ms', 'WiFi_Latency_ms', 'Total_Latency_ms']:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

    except FileNotFoundError:
        print(f"❌ Error: File {CSV_FILE_PATH} tidak ditemukan. Pastikan path benar.")
        return
    except Exception as e:
        print(f"❌ Error saat membaca CSV: {e}")
        return

    if total_samples == 0:
         print("⚠️ Data CSV kosong. Tidak ada yang bisa diproses.")
         return

    # ====================================================================
    # BAGIAN 1: STATISTIK GENERAL
    # ====================================================================
    print("   [Processing] Menghitung statistik general...")
    stats = {}
    stats['FPS_min'] = df['FPS'].min()
    stats['FPS_max'] = df['FPS'].max()
    stats['FPS_mean'] = df['FPS'].mean()
    stats['FPS_std'] = df['FPS'].std()

    for col in ['Edge_Latency_ms', 'WiFi_Latency_ms', 'Total_Latency_ms']:
        stats[f'{col}_min'] = df[col].min()
        stats[f'{col}_max'] = df[col].max()
        stats[f'{col}_mean'] = df[col].mean()
        stats[f'{col}_std'] = df[col].std()

    total_mean = stats['Total_Latency_ms_mean']
    if total_mean > 0:
        edge_pct = (stats['Edge_Latency_ms_mean'] / total_mean) * 100
        wifi_pct = (stats['WiFi_Latency_ms_mean'] / total_mean) * 100
    else:
        edge_pct = 0
        wifi_pct = 0

    if stats['Edge_Latency_ms_mean'] > 0:
        theoretical_fps = 1000 / stats['Edge_Latency_ms_mean']
    else:
        theoretical_fps = 0

    # ====================================================================
    # BAGIAN 2: FORMATTING KE LATEX GENERAL
    # ====================================================================
    tex_content_general = f"""% File ini digenerate otomatis oleh process_latency_data.py
% JANGAN DIEDIT MANUAL. Update file CSV-nya, lalu jalankan script Python-nya lagi.
% Waktu generate: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

% ==================================================
% METADATA PENGUJIAN TOTAL
% ==================================================
\\newcommand{{\\TotalSamples}}{{{total_samples}}}

% ==================================================
% RANGKUMAN GENERAL: KINERJA RASPI (BAB 4.5)
% ==================================================
\\newcommand{{\\FPSMin}}{{{fmt(stats['FPS_min'])}}}
\\newcommand{{\\FPSMax}}{{{fmt(stats['FPS_max'])}}}
\\newcommand{{\\FPSMean}}{{{fmt(stats['FPS_mean'])}}}
\\newcommand{{\\FPSStd}}{{{fmt(stats['FPS_std'])}}}
\\newcommand{{\\TheoreticalFPS}}{{{fmt(theoretical_fps)}}}

% --- Statistik Latensi Pemrosesan Edge (T_edge) ---
\\newcommand{{\\EdgeMin}}{{{fmt(stats['Edge_Latency_ms_min'])}}}
\\newcommand{{\\EdgeMax}}{{{fmt(stats['Edge_Latency_ms_max'])}}}
\\newcommand{{\\EdgeMean}}{{{fmt(stats['Edge_Latency_ms_mean'])}}}
\\newcommand{{\\EdgeStd}}{{{fmt(stats['Edge_Latency_ms_std'])}}}

% ==================================================
% RANGKUMAN GENERAL: ANALISIS LATENSI TOTAL (BAB 4.6)
% ==================================================
% --- Statistik Latensi Jaringan Nirkabel (T_wifi) ---
\\newcommand{{\\WiFiMin}}{{{fmt(stats['WiFi_Latency_ms_min'])}}}
\\newcommand{{\\WiFiMax}}{{{fmt(stats['WiFi_Latency_ms_max'])}}}
\\newcommand{{\\WiFiMean}}{{{fmt(stats['WiFi_Latency_ms_mean'])}}}
\\newcommand{{\\WiFiStd}}{{{fmt(stats['WiFi_Latency_ms_std'])}}}

% --- Statistik Total Latensi (T_total) ---
\\newcommand{{\\TotalMin}}{{{fmt(stats['Total_Latency_ms_min'])}}}
\\newcommand{{\\TotalMax}}{{{fmt(stats['Total_Latency_ms_max'])}}}
\\newcommand{{\\TotalMean}}{{{fmt(stats['Total_Latency_ms_mean'])}}}
\\newcommand{{\\TotalStd}}{{{fmt(stats['Total_Latency_ms_std'])}}}

% --- Persentase Kontribusi Rata-rata ---
\\newcommand{{\\EdgePct}}{{{fmt(edge_pct)}}}
\\newcommand{{\\WiFiPct}}{{{fmt(wifi_pct)}}}
"""

    # ====================================================================
    # BAGIAN 3: GENERATE STATISTIK SPESIFIK PER SKENARIO (FITUR BARU)
    # ====================================================================
    tex_content_scenarios = generate_scenario_stats_latex(df)

    # ====================================================================
    # BAGIAN 4: GABUNGKAN DAN SIMPAN
    # ====================================================================
    full_tex_content = tex_content_general + tex_content_scenarios

    try:
        os.makedirs(os.path.dirname(TEX_OUTPUT_PATH), exist_ok=True)
        with open(TEX_OUTPUT_PATH, 'w') as f:
            f.write(full_tex_content)
        print(f"✅ Berhasil! Data statistik lengkap disimpan ke:\n{TEX_OUTPUT_PATH}")
        print("Cek file tersebut untuk melihat nama command baru yang aman untuk LaTeX.")
    except Exception as e:
        print(f"❌ Error saat menyimpan file .tex: {e}")

if __name__ == "__main__":
    process_data()