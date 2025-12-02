import numpy as np
import tensorflow as tf
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm # Library untuk progress bar (biar gak bosen nunggu)

# --- KONFIGURASI ---
TFLITE_MODEL_PATH = 'model.tflite' # Pastikan file ini ada di folder ini
DATA_PATH = os.path.join('Keypoints_Data') 

# Urutan label harus SAMA PERSIS dengan saat training
actions = np.array([
    'close_to_open_palm', 'open_to_close_palm', 
    'close_to_one', 'open_to_one', 
    'close_to_two', 'open_to_two', 
    'close_to_three', 'open_to_three', 
    'close_to_four', 'open_to_four'
])

label_map = {label:num for num, label in enumerate(actions)}

# --- 1. LOAD DATA ---
print("Memuat data tes...")
sequences, labels = [], []
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"Warning: Folder {action} tidak ditemukan!")
        continue
        
    files = os.listdir(action_path)
    for file in files:
        # Load file .npy
        res = np.load(os.path.join(action_path, file))
        sequences.append(res)
        labels.append(label_map[action])

X = np.array(sequences)
y_true = np.array(labels)
print(f"Total data dimuat: {len(X)} sampel.")

# --- 2. INISIALISASI TFLITE INTERPRETER ---
print(f"Memuat model TFLite: {TFLITE_MODEL_PATH}...")
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error memuat model: {e}")
    exit()

print("Mulai Inferensi TFLite...")

# --- 3. PREDIKSI (LOOPING) ---
y_pred = []

# Kita loop satu per satu karena TFLite interpreter biasanya input statis (1, 30, 63)
for sample in tqdm(X, desc="Evaluasi"):
    # Preprocessing: Tambah dimensi batch -> (1, 30, 63) dan pastikan float32
    input_data = np.expand_dims(sample, axis=0).astype(np.float32)
    
    # Set tensor input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Jalankan inferensi
    interpreter.invoke()
    
    # Ambil hasil output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Ambil index dengan probabilitas tertinggi (Argmax)
    predicted_class = np.argmax(output_data)
    y_pred.append(predicted_class)

y_pred = np.array(y_pred)

# --- 4. HITUNG METRIK ---
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Akurasi Model TFLite: {accuracy*100:.2f}%")

print("\n--- Laporan Klasifikasi ---")
print(classification_report(y_true, y_pred, target_names=actions))

# --- 5. VISUALISASI CONFUSION MATRIX ---
print("Membuat grafik Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=actions, yticklabels=actions, cmap='Blues')
plt.ylabel('Label Asli (Ground Truth)')
plt.xlabel('Prediksi Model (TFLite)')
plt.title(f'Confusion Matrix - Model TFLite (Akurasi: {accuracy*100:.2f}%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Simpan gambar
output_filename = 'confusion_matrix_tflite.png'
plt.savefig(output_filename)
print(f"✅ Gambar tersimpan: {output_filename}")
# plt.show() # Uncomment jika dijalankan di laptop yang ada layarnya