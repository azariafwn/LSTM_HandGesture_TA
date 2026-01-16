import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Ambil folder tempat script ini berada
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Naik satu level dari folder 'LSTM_HandGesture' ke 'code_gesture'
# Lalu naik lagi ke 'PRATA' (sesuai struktur folder kamu)
# Lalu masuk ke 'BukuTATekkomLatex'
# Sesuaikan '..' sebanyak yang dibutuhkan untuk keluar dari folder coding ke folder root project
LATEX_PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../../BukuTATekkomLatex'))

TEX_DATA_DIR = os.path.join(LATEX_PROJECT_DIR, 'data') 
IMG_DIR = os.path.join(LATEX_PROJECT_DIR, 'gambar') # Tambahkan ini untuk gambar grafik

# Buat folder jika belum ada
os.makedirs(TEX_DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


total_start = time.time()

# Arahkan path ke folder Keypoints_Data
DATA_PATH = os.path.join('Keypoints_Data') 

actions = np.array(['close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three', 'close_to_four', 'open_to_four'])
sequence_length = 30
# --------------------------------------------------------

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    # --- Memuat semua file .npy yang ada secara dinamis ---
    action_path = os.path.join(DATA_PATH, action)
    sequence_files = os.listdir(action_path)
    
    print(f'Memuat {len(sequence_files)} sampel data untuk gestur "{action}"...')
    
    for sequence_file in sequence_files:
        res = np.load(os.path.join(action_path, sequence_file))
        sequences.append(res)
        labels.append(label_map[action])

print(f"\nTotal data yang dimuat: {len(sequences)} video.")

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Bagi data menjadi data latih (train) dan data uji (test)
# Menggunakan 20% data untuk testing
test_size=0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

# # --- MEMBANGUN ARSITEKTUR MODEL LSTM DENGAN DROPOUT ---  
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(sequence_length, 21*3)))
model.add(Dropout(0.5)) # <-- Tambahan Dropout
model.add(LSTM(128, return_sequences=True, activation='tanh'))
model.add(Dropout(0.5)) # <-- Tambahan Dropout
model.add(LSTM(64, return_sequences=False, activation='tanh'))
model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Kompilasi model
optimizer='Adam'
model_loss='categorical_crossentropy'
model.compile(optimizer=optimizer, loss=model_loss, metrics=['categorical_accuracy'])

# Ringkasan model
model.summary()

# --- PERSIAPAN CALLBACKS ---
# Hentikan training jika 'val_loss' tidak membaik selama 20 epoch
patience=20
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
# Simpan model terbaik berdasarkan 'val_loss'
model_checkpoint = ModelCheckpoint(
    'hand_gesture_model_terbaik.keras',  # Simpan dengan format .keras
    monitor='val_loss', 
    save_best_only=True
)

print("\nMulai Training Model...")
training_start = time.time()

# --- PELATIHAN MODEL ---
# Gunakan X_test dan y_test sebagai data validasi
# Model akan dievaluasi pada data ini di setiap akhir epoch
max_epochs = 200
batch_size = 32
history = model.fit(
    X_train, 
    y_train, 
    epochs=max_epochs, # Epochs bisa dibuat lebih banyak karena ada EarlyStopping
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint]
)

training_end = time.time() # Catat waktu selesai training
print(f"Training Selesai dalam: {training_end - training_start:.2f} detik.")

# --- SIMPAN MODEL FINAL (OPSIONAL, KARENA SUDAH DISIMPAN OLEH CHECKPOINT) ---
model.save('hand_gesture_model_final.keras')
print("Model final berhasil disimpan.")

# --- EVALUASI MODEL DENGAN BOBOT TERBAIK ---
# Model sudah otomatis dikembalikan ke bobot terbaik berkat 'restore_best_weights=True'
print("\nMengevaluasi model dengan bobot terbaik dari data tes...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Akurasi pada data tes: {accuracy*100:.2f}%")

# --- Simpan akurasi ke file ---
with open("latest_accuracy.txt", "w") as f:
    f.write(f"{accuracy*100:.2f}")
# -----------------------------------------------

total_end = time.time()
print(f"\nTotal Waktu Eksekusi Script: {total_end - total_start:.2f} detik.")

# ==========================================
# --- BAGIAN VISUALISASI GRAFIK (BARU) ---
# ==========================================
print("\nMembuat grafik riwayat training...")

# Ambil data dari history
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1) # Membuat sumbu X sesuai jumlah epoch asli

# --- GRAFIK 1: AKURASI ---
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Grafik Akurasi Model (Training vs Validation)')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(os.path.join(IMG_DIR, 'grafik_akurasi.png')) 
print(f"✅ Grafik Akurasi disimpan di: {os.path.join(IMG_DIR, 'grafik_akurasi.png')}")
plt.close() # Tutup plot agar tidak menumpuk

# --- GRAFIK 2: LOSS ---
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Grafik Loss Model (Training vs Validation)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(IMG_DIR, 'grafik_loss.png'))
print(f"✅ Grafik Loss disimpan di: {os.path.join(IMG_DIR, 'grafik_loss.png')}")
plt.close()




# ==========================================
# --- BAGIAN AUTO-GENERATE CONFIG TABLE LATEX ---
# ==========================================
print("\nMenyimpan konfigurasi pelatihan ke file LaTeX...")


OUTPUT_CONFIG_FILE = os.path.join(TEX_DATA_DIR, 'training_config.tex')

# --- AMBIL DATA REAL DARI VARIABEL DI ATAS ---
# 1. Optimizer: Ambil langsung dari object model biar akurat
real_optimizer = model.optimizer.get_config()['name'] 

# 2. Learning Rate: Coba ambil dari optimizer yang aktif
real_lr = model.optimizer.learning_rate.numpy()

# Ambil nama loss, pastikan string, dan ganti '_' dengan '\_' agar aman di LaTeX
raw_loss = model_loss
if hasattr(raw_loss, '__name__'): # Jika loss berupa object fungsi
    raw_loss = raw_loss.__name__

# Escape underscore untuk LaTeX (categorical_crossentropy -> categorical\_crossentropy)
safe_loss_fn = str(raw_loss).replace('_', '\\_') 
# ---------------------------

# 3. Split Ratio: Hitung persentase real
real_split_val = int(test_size * 100)
real_split_train = 100 - real_split_val

# 4. Buat Konten LaTeX
tex_content = f"""% Data Konfigurasi Training Otomatis
% Diambil langsung dari variabel eksekusi Python
% Tanggal: {time.strftime("%Y-%m-%d %H:%M:%S")}

\\newcommand{{\\TrainOptimizer}}{{{real_optimizer}}}
\\newcommand{{\\TrainLearningRate}}{{{real_lr}}}
\\newcommand{{\\TrainLossFn}}{{{safe_loss_fn}}}
\\newcommand{{\\TrainBatchSize}}{{{batch_size}}}  % Mengambil variabel BATCH_SIZE
\\newcommand{{\\TrainMaxEpochs}}{{{max_epochs}}}  % Mengambil variabel MAX_EPOCHS
\\newcommand{{\\TrainPatience}}{{{patience}}}      % Mengambil variabel PATIENCE
\\newcommand{{\\TrainSplitTrain}}{{{real_split_train}}} % Hasil hitungan
\\newcommand{{\\TrainSplitVal}}{{{real_split_val}}}     % Hasil hitungan
"""

with open(OUTPUT_CONFIG_FILE, "w") as f:
    f.write(tex_content)

print(f"✅ Konfigurasi tersimpan di: {OUTPUT_CONFIG_FILE}")


# ... (Kode sebelumnya: plt.savefig dan bagian export config awal) ...

# ==========================================
# --- BAGIAN AUTO-GENERATE ARSITEKTUR KE LATEX ---
# ==========================================

# Kita ambil info langsung dari layer model biar AKURAT 100%
# Struktur di kode Anda:
# Layer 0: LSTM
# Layer 1: Dropout
# Layer 2: LSTM
# Layer 3: Dropout
# Layer 4: LSTM
# Layer 5: Dense
# Layer 6: Dense
# Layer 7: Output

# Ambil data unit/rate dari layer
lstm1_units = model.layers[0].units
drop1_rate  = model.layers[1].rate
lstm2_units = model.layers[2].units
drop2_rate  = model.layers[3].rate
lstm3_units = model.layers[4].units
dense1_units = model.layers[5].units
dense2_units = model.layers[6].units
output_units = model.layers[7].units

# Update konten LaTeX (Tambahkan variabel arsitektur)
# Kita append (tambahkan) ke string tex_content yang sudah ada sebelumnya
tex_content += f"""
% --- ARSITEKTUR MODEL DINAMIS ---
\\newcommand{{\\TrainSeqLength}}{{{sequence_length}}}
\\newcommand{{\\TrainInputFeatures}}{{63}} % 21 titik x 3 dimensi

\\newcommand{{\\LstmOneUnits}}{{{lstm1_units}}}
\\newcommand{{\\DropOneRate}}{{{drop1_rate}}}
\\newcommand{{\\LstmTwoUnits}}{{{lstm2_units}}}
\\newcommand{{\\DropTwoRate}}{{{drop2_rate}}}
\\newcommand{{\\LstmThreeUnits}}{{{lstm3_units}}}

\\newcommand{{\\DenseOneUnits}}{{{dense1_units}}}
\\newcommand{{\\DenseTwoUnits}}{{{dense2_units}}}
\\newcommand{{\\OutputUnits}}{{{output_units}}}
"""

# Tulis ulang file config dengan tambahan baru
with open(OUTPUT_CONFIG_FILE, "w") as f:
    f.write(tex_content)

print(f"✅ Konfigurasi & Arsitektur tersimpan di: {OUTPUT_CONFIG_FILE}")