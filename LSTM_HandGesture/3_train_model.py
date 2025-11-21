import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Arahkan path ke folder Keypoints_Data
DATA_PATH = os.path.join('Keypoints_Data') 

actions = np.array(['close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three'])
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

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
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Ringkasan model
model.summary()

# --- PERSIAPAN CALLBACKS ---
# Hentikan training jika 'val_loss' tidak membaik selama 20 epoch
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
# Simpan model terbaik berdasarkan 'val_loss'
model_checkpoint = ModelCheckpoint(
    'hand_gesture_model_terbaik.keras',  # Simpan dengan format .keras
    monitor='val_loss', 
    save_best_only=True
)

# --- PELATIHAN MODEL ---
# Gunakan X_test dan y_test sebagai data validasi
# Model akan dievaluasi pada data ini di setiap akhir epoch
history = model.fit(
    X_train, 
    y_train, 
    epochs=200, # Epochs bisa dibuat lebih banyak karena ada EarlyStopping
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint]
)

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