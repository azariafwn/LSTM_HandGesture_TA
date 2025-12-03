import tensorflow as tf

# --- Baca akurasi dari file ---
try:
    with open("latest_accuracy.txt", "r") as f:
        accuracy_str = f.read()
except FileNotFoundError:
    accuracy_str = "N/A"
# -----------------------------------------------

# Tentukan spesifikasi input STATIS (Batch size=1)
INPUT_SHAPE = (1, 30, 63) 
# -------------------------------------

# Muat model .keras terbaik
model = tf.keras.models.load_model('hand_gesture_model_terbaik.keras')
print("Model .keras berhasil dimuat.")

# --- METODE KONVERSI BARU YANG LEBIH ROBUST ---
run_model = tf.function(model.call)
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(INPUT_SHAPE, model.inputs[0].dtype)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func],
    model
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]
# -----------------------------------------------

print("Memulai konversi TFLite murni (metode concrete function + batch size statis)...")

# Lakukan konversi
tflite_model = converter.convert()

# Simpan model .tflite
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("\n-------------------------------------------------")
print("BERHASIL!")
print(f"Model TFLite MURNI (Akurasi {accuracy_str}%) berhasil dibuat.")
print("File 'model.tflite' ini sekarang 100% siap untuk Raspberry Pi.")
print("-------------------------------------------------")

import os

# ================= KONFIGURASI PATH =================
# Ganti path ini agar mengarah ke folder root proyek LaTeX Anda
LATEX_PROJECT_PATH = r"C:/zafaa/kuliah/SEMESTER7/PRATA/BukuTATekkomLatex/data"

# Nama file model di folder Python saat ini
H5_MODEL_NAME = 'hand_gesture_model_terbaik.keras'
TFLITE_MODEL_NAME = 'model.tflite'

# Nama file data yang akan dibuat di folder LaTeX
OUTPUT_TEX_FILE = 'model_sizes_data.tex'
# ====================================================

def get_file_size_kb(file_path):
    """Mendapatkan ukuran file dalam KB (dibulatkan ke integer terdekat)."""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} tidak ditemukan!")
        return 0
    # Dapatkan ukuran dalam bytes, bagi 1024 untuk KB, bulatkan
    return round(os.path.getsize(file_path) / 1024)

print("\n--- Memulai Update Data Ukuran Model untuk LaTeX ---")

# 1. Baca ukuran file
h5_size_kb = get_file_size_kb(H5_MODEL_NAME)
tflite_size_kb = get_file_size_kb(TFLITE_MODEL_NAME)

print(f"Ukuran {H5_MODEL_NAME}: {h5_size_kb} KB")
print(f"Ukuran {TFLITE_MODEL_NAME}: {tflite_size_kb} KB")

# 2. Hitung Persentase Reduksi
if h5_size_kb > 0:
    # Rumus: (Awal - Akhir) / Awal * 100
    reduction_pct = ((h5_size_kb - tflite_size_kb) / h5_size_kb) * 100
    # Format menjadi string dengan 1 angka di belakang koma (contoh: "89.6")
    reduction_pct_str = f"{reduction_pct:.1f}"
else:
    reduction_pct_str = "0.0"

print(f"Reduksi dihitung: {reduction_pct_str}%")

# 3. Siapkan konten file .tex
# Kita membuat command LaTeX baru: \SizeHfive, \SizeTflite, dan \ReductionPct
tex_content = f"""% File ini digenerate otomatis oleh script Python.
% JANGAN DIEDIT MANUAL. Jalankan script Python untuk update data.

% Mendefinisikan command baru untuk data ukuran model
\\newcommand{{\\SizeHfive}}{{{h5_size_kb}}}
\\newcommand{{\\SizeTflite}}{{{tflite_size_kb}}}
\\newcommand{{\\ReductionPct}}{{{reduction_pct_str}}}
"""

# 4. Tentukan lokasi penyimpanan file .tex target
output_path = os.path.join(LATEX_PROJECT_PATH, OUTPUT_TEX_FILE)

# 5. Tulis file .tex
try:
    with open(output_path, "w") as f:
        f.write(tex_content)
    print(f"\nSUKSES! Data telah ditulis ke: {output_path}")
    print("Silakan compile ulang dokumen LaTeX Anda.")
except Exception as e:
    print(f"\nERROR saat menulis file ke folder LaTeX: {e}")
    print("Pastikan LATEX_PROJECT_PATH sudah benar.")

print("----------------------------------------------------")