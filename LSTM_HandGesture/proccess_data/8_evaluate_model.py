import os
import time
import psutil
import gc
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from tqdm import tqdm

# === KONFIGURASI PATH DAN NAMA FILE ===
DATA_PATH = os.path.join('Keypoints_Data') 
KERAS_MODEL_PATH = 'hand_gesture_model_terbaik.keras'
TFLITE_MODEL_PATH = 'model.tflite'

# --- [KONFIGURASI OUTPUT LATEX] ---
LATEX_PROJECT_DIR = 'C:/zafaa/kuliah/SEMESTER7/PRATA/BukuTATekkomLatex' 
IMG_DIR = os.path.join(LATEX_PROJECT_DIR, 'gambar/')
CM_KERAS_IMG = os.path.join(IMG_DIR, 'confusion_matrix_keras.png')
CM_TFLITE_IMG = os.path.join(IMG_DIR, 'confusion_matrix_tflite.png')
TEX_DATA_DIR = os.path.join(LATEX_PROJECT_DIR, 'data/') 
OUTPUT_TEX_FILE = os.path.join(TEX_DATA_DIR, 'model_benchmark_data.tex')

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TEX_DATA_DIR, exist_ok=True)

# Nama Label harus SAMA PERSIS dengan folder/training
LABELS = [
    'close_to_open_palm', 'open_to_close_palm', 
    'close_to_one', 'open_to_one', 
    'close_to_two', 'open_to_two', 
    'close_to_three', 'open_to_three', 
    'close_to_four', 'open_to_four'
]

# === FUNGSI HELPER ===
def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_file_size_kb(file_path):
    if not os.path.exists(file_path): return 0
    return round(os.path.getsize(file_path) / 1024)

def plot_and_save_cm(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
    plt.title(title)
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Label Prediksi')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Gambar CM tersimpan di: {save_path}")

# === 1. LOAD DAN SPLIT DATA ===
print("🔄 Memuat dataset...")
sequences, labels = [], []
for label_idx, label_name in enumerate(LABELS):
    label_path = os.path.join(DATA_PATH, label_name)
    if not os.path.exists(label_path): continue
    for file_name in os.listdir(label_path):
        if file_name.endswith('.npy'):
            try:
                data = np.load(os.path.join(label_path, file_name))
                if data.shape == (30, 63):
                    sequences.append(data)
                    labels.append(label_idx)
            except: pass

X = np.array(sequences, dtype=np.float32)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Data Test siap: {len(X_test)} sampel.")

dummy_input = np.zeros((1, 30, 63), dtype=np.float32)
results = {}

# === 2. EVALUASI MODEL KERAS (.H5) ===
print("\n--- 🧠 Memulai Evaluasi Model KERAS (.h5) ---")
gc.collect()
base_mem_keras = get_memory_usage_mb()

try:
    model_keras = tf.keras.models.load_model(KERAS_MODEL_PATH)
    model_keras.predict(dummy_input, verbose=0) # Warmup
    peak_mem_keras = get_memory_usage_mb()
    results['ram_keras'] = peak_mem_keras - base_mem_keras

    start_time = time.time()
    y_pred_keras_probs = model_keras.predict(X_test, verbose=0)
    end_time = time.time()
    
    results['latency_keras'] = ((end_time - start_time) / len(X_test)) * 1000
    y_pred_keras = np.argmax(y_pred_keras_probs, axis=1)
    
    # Hitung Metrik Lengkap (Weighted)
    results['acc_keras'] = accuracy_score(y_test, y_pred_keras) * 100
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred_keras, average='weighted')
    results['prec_keras'] = p * 100
    results['recall_keras'] = r * 100
    results['f1_keras'] = f1 * 100

    print(f"Akurasi Keras: {results['acc_keras']:.2f}%")
    plot_and_save_cm(y_test, y_pred_keras, 'Confusion Matrix - Model Keras', CM_KERAS_IMG)

except Exception as e: print(f"❌ Error Keras: {e}")

del model_keras
tf.keras.backend.clear_session()
gc.collect()

# === 3. EVALUASI MODEL TFLITE (.tflite) ===
print("\n--- ⚡ Memulai Evaluasi Model TFLITE (.tflite) ---")
gc.collect()
base_mem_tflite = get_memory_usage_mb()

try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_idx, output_idx = input_details[0]['index'], output_details[0]['index']

    interpreter.set_tensor(input_idx, dummy_input)
    interpreter.invoke() # Warmup
    peak_mem_tflite = get_memory_usage_mb()
    results['ram_tflite'] = peak_mem_tflite - base_mem_tflite

    y_pred_tflite = []
    start_time = time.time()
    for i in tqdm(range(len(X_test)), desc="Inferensi TFLite"):
        sample = np.expand_dims(X_test[i], axis=0)
        interpreter.set_tensor(input_idx, sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_idx)
        y_pred_tflite.append(np.argmax(output[0]))
    end_time = time.time()

    results['latency_tflite'] = ((end_time - start_time) / len(X_test)) * 1000
    
    # Hitung Metrik Global (Weighted)
    results['acc_tflite'] = accuracy_score(y_test, y_pred_tflite) * 100
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred_tflite, average='weighted')
    results['prec_tflite'] = p * 100
    results['recall_tflite'] = r * 100
    results['f1_tflite'] = f1 * 100
    
    # Hitung Metrik PER KELAS (Average=None) untuk Tabel Detail
    p_class, r_class, f1_class, _ = precision_recall_fscore_support(y_test, y_pred_tflite, average=None)

    print(f"Akurasi TFLite: {results['acc_tflite']:.2f}%")
    plot_and_save_cm(y_test, y_pred_tflite, 'Confusion Matrix - Model TFLite', CM_TFLITE_IMG)

except Exception as e: print(f"❌ Error TFLite: {e}")

# === 4. GENERATE DATA TEXT UNTUK LATEX ===
results['size_keras_kb'] = get_file_size_kb(KERAS_MODEL_PATH)
results['size_tflite_kb'] = get_file_size_kb(TFLITE_MODEL_PATH)
results['size_reduction'] = ((results['size_keras_kb'] - results['size_tflite_kb']) / results['size_keras_kb']) * 100

if results['ram_keras'] > 0:
    results['ram_reduction'] = ((results['ram_keras'] - results['ram_tflite']) / results['ram_keras']) * 100
else:
    results['ram_reduction'] = 0
# -------------------------

tex_content = f"""% Data Benchmark Otomatis
% Tanggal: {time.strftime("%Y-%m-%d %H:%M:%S")}

% --- SUMBER DAYA ---
\\newcommand{{\\SizeKerasKB}}{{{results['size_keras_kb']}}}
\\newcommand{{\\SizeTfliteKB}}{{{results['size_tflite_kb']}}}
\\newcommand{{\\SizeReductionPct}}{{{results['size_reduction']:.1f}}}
\\newcommand{{\\RamKerasMB}}{{{results['ram_keras']:.2f}}}
\\newcommand{{\\RamTfliteMB}}{{{results['ram_tflite']:.2f}}}
\\newcommand{{\\RamReductionPct}}{{{results['ram_reduction']:.1f}}}

% --- KINERJA GLOBAL ---
\\newcommand{{\\AccKeras}}{{{results['acc_keras']:.2f}}}
\\newcommand{{\\PrecKeras}}{{{results['prec_keras']:.2f}}}
\\newcommand{{\\RecallKeras}}{{{results['recall_keras']:.2f}}}
\\newcommand{{\\FScoreKeras}}{{{results['f1_keras']:.2f}}}

\\newcommand{{\\AccTflite}}{{{results['acc_tflite']:.2f}}}
\\newcommand{{\\PrecTflite}}{{{results['prec_tflite']:.2f}}}
\\newcommand{{\\RecallTflite}}{{{results['recall_tflite']:.2f}}}
\\newcommand{{\\FScoreTflite}}{{{results['f1_tflite']:.2f}}}

% --- LATENSI ---
\\newcommand{{\\LatencyKerasPC}}{{{results['latency_keras']:.2f}}}
\\newcommand{{\\LatencyTflitePC}}{{{results['latency_tflite']:.2f}}}

% --- KINERJA DETAIL PER KELAS (TFLITE) ---
"""

# Loop untuk membuat macro per kelas (Clean nama agar valid LaTeX)
# pake CamelCase
for i, label in enumerate(LABELS):
    clean_label = label.replace('_', ' ').title().replace(' ', '')
    
    tex_content += f"\\newcommand{{\\Prec{clean_label}}}{{{p_class[i]:.2f}}}\n"
    tex_content += f"\\newcommand{{\\Rec{clean_label}}}{{{r_class[i]:.2f}}}\n"
    tex_content += f"\\newcommand{{\\FScore{clean_label}}}{{{f1_class[i]:.2f}}}\n"

with open(OUTPUT_TEX_FILE, "w") as f:
    f.write(tex_content)
print(f"\n💾 Data benchmark LENGKAP tersimpan di: {OUTPUT_TEX_FILE}")