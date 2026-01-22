import pandas as pd
import os
import numpy as np

# === KONFIGURASI ===
# Path ke file CSV input
CSV_FILE_PATH = 'C:/zafaa/kuliah/SEMESTER7/PRATA/code_gesture/LSTM_HandGesture/data/data_pengujian.csv'
# Path ke file TEX output yang akan dibuat
LATEX_PROJECT_DIR = 'C:/zafaa/kuliah/SEMESTER7/PRATA/BukuTATekkomLatex'
TEX_DATA_DIR = os.path.join(LATEX_PROJECT_DIR, 'data/')
TEX_OUTPUT_PATH = os.path.join(TEX_DATA_DIR, 'latency_pengujian_stats.tex')

def process_data():
    print(f"--- Memulai pemrosesan data dari {CSV_FILE_PATH} ---")

    # 1. Baca data dari CSV
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        total_samples = len(df)
        print(f"Berhasil membaca {total_samples} baris data.")
    except FileNotFoundError:
        print(f"Error: File {CSV_FILE_PATH} tidak ditemukan. Pastikan path benar.")
        return

    # 2. Hitung Statistik
    stats = {}

    # --- A. Statistik FPS (Untuk Bab 4.5) ---
    stats['FPS_min'] = df['FPS'].min()
    stats['FPS_max'] = df['FPS'].max()
    stats['FPS_mean'] = df['FPS'].mean()
    stats['FPS_std'] = df['FPS'].std()

    # --- B. Statistik Latensi (Untuk Bab 4.5 & 4.6) ---
    # Loop ini menghitung statistik lengkap untuk Edge, WiFi, dan Total
    for col in ['Edge_Latency_ms', 'WiFi_Latency_ms', 'Total_Latency_ms']:
        stats[f'{col}_min'] = df[col].min()
        stats[f'{col}_max'] = df[col].max()
        stats[f'{col}_mean'] = df[col].mean()
        stats[f'{col}_std'] = df[col].std() # Std dev dihitung untuk semua di sini

    # 3. Hitung Persentase Kontribusi (Untuk Bab 4.6)
    total_mean = stats['Total_Latency_ms_mean']
    if total_mean > 0:
        edge_pct = (stats['Edge_Latency_ms_mean'] / total_mean) * 100
        wifi_pct = (stats['WiFi_Latency_ms_mean'] / total_mean) * 100
    else:
        edge_pct = 0
        wifi_pct = 0

    # 4. Hitung nilai ilustrasi teoritis FPS (Untuk Bab 4.5)
    if stats['Edge_Latency_ms_mean'] > 0:
        theoretical_fps = 1000 / stats['Edge_Latency_ms_mean']
    else:
        theoretical_fps = 0

    print("Statistik berhasil dihitung.")

    # --- FORMATTING KE LATEX ---
    def fmt(val):
        # Handle jika nilai NaN (misal data kosong)
        if pd.isna(val) or np.isnan(val):
             return "0.00"
        return f"{val:.2f}"

    # Buat konten file .tex
    tex_content = f"""% File ini digenerate otomatis oleh process_latency_data.py
% JANGAN DIEDIT MANUAL. Update file CSV-nya, lalu jalankan script Python-nya lagi.
% Waktu generate: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

% ==================================================
% METADATA PENGUJIAN
% ==================================================
\\newcommand{{\\TotalSamples}}{{{total_samples}}}

% ==================================================
% DATA UNTUK BAB 4.5 (KINERJA RASPI)
% ==================================================

% --- Statistik FPS ---
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
% DATA UNTUK BAB 4.6 (ANALISIS LATENSI TOTAL)
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

    # 5. Simpan ke file .tex output
    try:
        os.makedirs(os.path.dirname(TEX_OUTPUT_PATH), exist_ok=True)
        with open(TEX_OUTPUT_PATH, 'w') as f:
            f.write(tex_content)
        print(f"✅ Berhasil! Data statistik disimpan ke {TEX_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saat menyimpan file .tex: {e}")

if __name__ == "__main__":
    process_data()