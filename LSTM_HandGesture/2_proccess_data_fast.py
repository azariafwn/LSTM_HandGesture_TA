import cv2
import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor

# --- PENGATURAN ---
DATA_PATH = os.path.join('MP_Data')
KEYPOINTS_PATH = os.path.join('Keypoints_Data')
actions = np.array(['close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three'])
sequence_length = 30

# --- FUNGSI WORKER (Modifikasi: Import dipindah ke sini) ---
def process_single_sequence(args):
    # !!! IMPORT DI DALAM FUNGSI UNTUK MENGHINDARI CRASH DI WINDOWS !!!
    import mediapipe as mp
    
    mp_holistic = mp.solutions.holistic
    action, sequence, full_action_path, save_path = args
    
    # Gunakan 'static_image_mode=True' untuk akurasi lebih baik pada gambar lepasan
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=True) as holistic:
        window = []
        valid_sequence = True
        
        for frame_num in range(sequence_length):
            img_path = os.path.join(full_action_path, sequence, f"{frame_num}.jpg")
            frame = cv2.imread(img_path)
            
            if frame is None:
                valid_sequence = False
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            
            # Extract Keypoints (Logic disalin ke dalam sini biar aman)
            if results.right_hand_landmarks:
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
            else:
                rh = np.zeros(21*3)
            
            window.append(rh)
            
        if valid_sequence and len(window) == sequence_length:
            npy_path = os.path.join(save_path, sequence)
            np.save(npy_path, window)
            return f"✅ {action}/{sequence} OK"
        else:
            return f"❌ {action}/{sequence} GAGAL (Frame error)"

def main():
    # Fix untuk Windows agar tidak berebut resource saat start
    # Kita batasi worker agar tidak membuat PC freeze total jika RAM ngepas
    # os.cpu_count() - 2 artinya menyisakan 2 core untuk OS agar tidak lag
    max_workers = 4 

    print("Memvalidasi folder...")
    tasks = []
    
    for action in actions:
        save_path = os.path.join(KEYPOINTS_PATH, action)
        os.makedirs(save_path, exist_ok=True)
        
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path): continue
        
        sequences = [d for d in os.listdir(action_path) if d.isdigit()]
        
        for seq in sequences:
            tasks.append((action, seq, action_path, save_path))

    total_files = len(tasks)
    print(f"Total ada {total_files} video sequence.")
    print(f"Mulai Multiprocessing (Max Workers: {max_workers})...")
    
    start_time = time.time()
    
    # Jalankan executor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_sequence, tasks))
        
    end_time = time.time()
    
    success_count = sum(1 for r in results if "OK" in r)
    print(f"\nSelesai! {success_count}/{total_files} berhasil.")
    print(f"Waktu: {end_time - start_time:.2f} detik.")

if __name__ == '__main__':
    main()