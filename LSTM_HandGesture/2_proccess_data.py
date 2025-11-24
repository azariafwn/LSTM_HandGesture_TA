import cv2
import numpy as np
import os
import shutil 
import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return rh

# --- PENGATURAN ---
DATA_PATH = os.path.join('MP_Data')
KEYPOINTS_PATH = os.path.join('Keypoints_Data')
# actions = np.array(['thumbs_down_to_up', 'thumbs_up_to_down','close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'close_to_two'])
actions = np.array(['close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three'])
sequence_length = 30
# --------------------

# --- FUNGSI VALIDASI DATA ---
def validate_data():
    print("Memulai validasi data mentah...")
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path): continue
        
        sequences = [d for d in os.listdir(action_path) if d.isdigit()]
        for sequence in sequences:
            sequence_path = os.path.join(action_path, sequence)
            num_frames = len(os.listdir(sequence_path))
            if num_frames != sequence_length:
                print(f'  -> Menghapus data tidak lengkap: {sequence_path} (hanya ada {num_frames} frame)')
                shutil.rmtree(sequence_path) # Hapus folder yang tidak lengkap
    print("Validasi selesai.\n")
# ---------------------------------------------

# Jalankan validasi sebelum memproses
validate_data()

# Membuat folder-folder baru di Keypoints_Data
for action in actions:
    os.makedirs(os.path.join(KEYPOINTS_PATH, action), exist_ok=True)

start_time = time.time()
print(f"Mulai memproses data pada: {time.ctime(start_time)}")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        sequences = [d for d in os.listdir(action_path) if d.isdigit()]
        
        print(f'Memproses {len(sequences)} video untuk gestur "{action}"...')

        for sequence in sequences:
            window = []
            for frame_num in range(sequence_length):
                img_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.jpg")
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Warning: Frame tidak ditemukan di {img_path}. Melanjutkan...")
                    continue
                
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                window.append(keypoints)
            
            # Hanya simpan jika panjang window sesuai
            if len(window) == sequence_length:
                npy_path = os.path.join(KEYPOINTS_PATH, action, str(sequence))
                np.save(npy_path, window)
            else:
                print(f"Skipping sequence {sequence} for action {action} due to incorrect frame count.")
            
end_time = time.time()
total_duration = end_time - start_time
print("\nEkstraksi fitur selesai untuk semua data yang valid.")
print(f"Total Waktu Proses: {total_duration:.2f} detik ({total_duration/60:.2f} menit).")