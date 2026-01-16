import cv2
import numpy as np
import os
import time
import mediapipe as mp

# --- TAMBAHAN 2: Inisialisasi MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# ----------------------------------------------

# --- PENGATURAN UTAMA (TETAP SAMA) ---
DATA_PATH = os.path.join('MP_Data') 
# actions = np.array(['close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three', 'close_to_four', 'open_to_four'])
# actions = np.array(['close_to_open_palm', 'open_to_close_palm'])
# actions = np.array(['close_to_one', 'open_to_one'])
# actions = np.array(['close_to_two', 'open_to_two'])
# actions = np.array(['close_to_three', 'open_to_three'])
actions = np.array(['close_to_four', 'open_to_four'])

no_sequences_to_add = 20
sequence_length = 30
# --- AKHIR PENGATURAN ---

os.makedirs(DATA_PATH, exist_ok=True)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Gunakan kamera default
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Gunakan kamera eksternal
# cap = cv2.VideoCapture(2, cv2.CAP_DSHOW) # Gunakan kamera eksternal
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# --- TAMBAHAN 3: Gunakan 'with' block untuk model MediaPipe ---
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        os.makedirs(action_path, exist_ok=True)
        
        dir_list = [d for d in os.listdir(action_path) if d.isdigit()]
        start_sequence = max([int(d) for d in dir_list]) + 1 if dir_list else 0
        
        print(f'Gestur "{action}": Data sudah ada {start_sequence} video. Merekam {no_sequences_to_add} video baru...')
        time.sleep(2)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # --- TAMBAHAN 4: Deteksi dan gambar landmark untuk visualisasi ---
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            debug_image = frame.copy() # Gambar di salinan agar frame asli tetap bersih
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # -------------------------------------------------------------
            
            cv2.putText(debug_image, f'SIAP MEREKAM GESTUR: "{action}"', (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(debug_image, "Tekan 'S' untuk Mulai...", (120, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Collect Gesture Data', debug_image) # Tampilkan gambar dengan landmark
            
            if cv2.waitKey(10) & 0xFF == ord('s'):
                break
                
        for sequence in range(start_sequence, start_sequence + no_sequences_to_add):
            sequence_path = os.path.join(action_path, str(sequence))
            os.makedirs(sequence_path, exist_ok=True)
            
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                # Ulangi visualisasi untuk hitung mundur
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                debug_image = frame.copy()
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                cv2.putText(debug_image, f'Video #{sequence}. Siap dalam {i} detik...', (15, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Collect Gesture Data', debug_image)
                cv2.waitKey(1000)

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret: break

                # Ulangi visualisasi saat merekam
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                debug_image = frame.copy()
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.putText(debug_image, f'MEREKAM GESTUR: {action} | Video: {sequence} | Frame: {frame_num}', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Collect Gesture Data', debug_image)
                
                # --- PENTING: Yang disimpan adalah 'frame' asli, bukan 'debug_image' ---
                frame_path = os.path.join(sequence_path, f"{frame_num}.jpg")
                cv2.imwrite(frame_path, frame)
                # --------------------------------------------------------------------

                if cv2.waitKey(10) & 0xFF == ord('q'): break
            
            if cv2.waitKey(10) & 0xFF == ord('q'): break
                
        if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()