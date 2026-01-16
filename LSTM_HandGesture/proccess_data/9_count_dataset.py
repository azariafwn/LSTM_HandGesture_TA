import os
import time

# === KONFIGURASI ===
# Path ke folder dataset (MP_Data)
DATA_PATH = os.path.join('MP_Data') 

# Path Output LaTeX
LATEX_PROJECT_DIR = 'C:\zafaa\kuliah\SEMESTER7\PRATA\BukuTATekkomLatex' 
TEX_DATA_DIR = os.path.join(LATEX_PROJECT_DIR, 'data/') 
OUTPUT_TEX_FILE = os.path.join(TEX_DATA_DIR, 'dataset_stats.tex')

# Pastikan folder tujuan ada
os.makedirs(TEX_DATA_DIR, exist_ok=True)

# Daftar Nama Folder Asli di MP_Data
# Kita mapping ke Logical Name untuk Laporan
GESTURE_MAPPING = {
    'close_to_open_palm': 'AksiOn',
    'open_to_close_palm': 'AksiOff',
    'close_to_one': 'P1_Close',
    'open_to_one':  'P1_Open',
    'close_to_two': 'P2_Close',
    'open_to_two':  'P2_Open',
    'close_to_three': 'P3_Close',
    'open_to_three':  'P3_Open',
    'close_to_four': 'P4_Close',
    'open_to_four':  'P4_Open'
}

print("🔄 Menghitung jumlah dataset...")

stats = {}
total_all = 0

for folder_name, logic_name in GESTURE_MAPPING.items():
    folder_path = os.path.join(DATA_PATH, folder_name)
    
    count = 0
    if os.path.exists(folder_path):
        # Hitung jumlah folder di dalamnya (setiap folder = 1 video sampel)
        # Filter agar hanya menghitung folder angka (bukan file sampah)
        try:
            items = os.listdir(folder_path)
            # Asumsi: setiap sampel adalah folder/file. 
            # Jika struktur MP_Data Anda: MP_Data/Action/0, MP_Data/Action/1, ...
            count = len(items) 
        except Exception as e:
            print(f"Error membaca {folder_name}: {e}")
    
    stats[logic_name] = count
    total_all += count
    print(f"   -> {folder_name}: {count} sampel")

# === HITUNG TOTAL PER KATEGORI (Opsional, jika tabel ingin digabung) ===
# Total Perangkat 1 (Close + Open)
total_p1 = stats.get('P1_Close', 0) + stats.get('P1_Open', 0)
total_p2 = stats.get('P2_Close', 0) + stats.get('P2_Open', 0)
total_p3 = stats.get('P3_Close', 0) + stats.get('P3_Open', 0)
total_p4 = stats.get('P4_Close', 0) + stats.get('P4_Open', 0)

# === GENERATE LATEX CONTENT ===
tex_content = f"""% Data Statistik Dataset Otomatis
% Tanggal Update: {time.strftime("%Y-%m-%d %H:%M:%S")}

% --- Rincian Per Folder Asli ---
\\newcommand{{\\CountAksiOn}}{{{stats['AksiOn']}}}
\\newcommand{{\\CountAksiOff}}{{{stats['AksiOff']}}}

\\newcommand{{\\CountP1Close}}{{{stats['P1_Close']}}}
\\newcommand{{\\CountP1Open}}{{{stats['P1_Open']}}}
\\newcommand{{\\CountP2Close}}{{{stats['P2_Close']}}}
\\newcommand{{\\CountP2Open}}{{{stats['P2_Open']}}}
\\newcommand{{\\CountP3Close}}{{{stats['P3_Close']}}}
\\newcommand{{\\CountP3Open}}{{{stats['P3_Open']}}}
\\newcommand{{\\CountP4Close}}{{{stats['P4_Close']}}}
\\newcommand{{\\CountP4Open}}{{{stats['P4_Open']}}}

% --- Rincian Gabungan (Total per Perangkat) ---
\\newcommand{{\\CountTotalP1}}{{{total_p1}}}
\\newcommand{{\\CountTotalP2}}{{{total_p2}}}
\\newcommand{{\\CountTotalP3}}{{{total_p3}}}
\\newcommand{{\\CountTotalP4}}{{{total_p4}}}

% --- Total Keseluruhan ---
\\newcommand{{\\CountTotalSemua}}{{{total_all}}}
"""

with open(OUTPUT_TEX_FILE, "w") as f:
    f.write(tex_content)

print(f"\n✅ Data statistik dataset tersimpan di: {OUTPUT_TEX_FILE}")
print(f"   Total Sampel: {total_all}")