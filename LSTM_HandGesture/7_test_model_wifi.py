import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
import time
import requests
from requests.exceptions import ConnectionError

# --- KONFIGURASI UTAMA ---
USE_VIDEO_FILE = False
VIDEO_PATH = 'video_testing.mp4'
COOLDOWN_DURATION = 1.5         # Jeda standar antar deteksi (dipercepat sedikit)
POST_COMMAND_COOLDOWN = 4.0     # [BARU] Jeda LEBIH LAMA setelah perintah sukses terkirim
ESP_IP = "10.141.159.103"       # Pastikan IP ini benar
BASE_URL = f"http://{ESP_IP}"
PREDICTION_THRESHOLD = 0.97
# -------------------------

print(f"Mencoba terhubung ke server ESP8266 di {BASE_URL}...")

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

# --- LOAD MODEL ---
TFLITE_MODEL_PATH = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model TFLite berhasil dimuat.")

def send_command(command):
    try:
        # Timeout sangat singkat agar video tidak lag
        response = requests.get(f"{BASE_URL}/{command}", timeout=0.5)
        if response.status_code == 200:
            print(f"BERHASIL MENGIRIM: Perintah {command}")
        else:
            print(f"Gagal mengirim: Status {response.status_code}")
    except (ConnectionError, requests.exceptions.Timeout):
        # Error koneksi wajar terjadi jika ESP sibuk/jauh, print saja lalu lanjut
        print(f"Peringatan: Gagal koneksi ke ESP ({command})") 

# --- DEFINISI GESTUR ---
actions = np.array([
    'close_to_open_palm', 'open_to_close_palm',
    'close_to_one', 'close_to_two',
    'open_to_one', 'open_to_two'
])

ACTION_GESTURES = ['close_to_open_palm', 'open_to_close_palm']
SELECTION_GESTURES = ['close_to_one', 'close_to_two', 'open_to_one', 'open_to_two']

sequence = []
current_action_state = None 
last_action_time = 0
STATE_TIMEOUT = 5 

# --- TIMER ---
last_valid_time = 0 
current_cooldown_limit = COOLDOWN_DURATION # Variabel dinamis untuk durasi cooldown
# -------------

# --- INISIALISASI INPUT ---
if USE_VIDEO_FILE:
    print(f"MODE: Menggunakan Video File ({VIDEO_PATH})")
    cap = cv2.VideoCapture(VIDEO_PATH)
else:
    print("MODE: Menggunakan Kamera Live")
    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    print("Error: Input tidak ditemukan.")
    exit()

prev_time = 0
fps = 0

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        curr_time = time.time()
        if prev_time > 0: fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        ret, frame = cap.read()
        if not ret:
            if USE_VIDEO_FILE:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        image, results = mediapipe_detection(frame, holistic)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        temp_action = '...'
        
        if len(sequence) == 30:
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            if np.max(prediction) > PREDICTION_THRESHOLD:
                temp_action = actions[np.argmax(prediction)]

        # =====================================================
        # LOGIKA STATE MACHINE + SMART COOLDOWN
        # =====================================================
        
        current_time_for_logic = time.time()
        final_command_sent = False 
        
        # Cek status Cooldown dengan limit yang dinamis
        time_since_last = current_time_for_logic - last_valid_time
        in_cooldown = time_since_last < current_cooldown_limit

        if temp_action != '...' and not in_cooldown:
            
            # 1. JIKA MENUNGGU SELEKSI PERANGKAT (State Aktif)
            if current_action_state is not None:
                if temp_action in SELECTION_GESTURES:
                    
                    # Eksekusi perintah
                    if temp_action in ['close_to_one', 'open_to_one']:
                        cmd = '11' if current_action_state == 'AKSI_ON' else '10'
                        print(f"PERINTAH: PERANGKAT 1 -> {current_action_state}")
                        send_command(cmd)
                    
                    elif temp_action in ['close_to_two', 'open_to_two']:
                        cmd = '21' if current_action_state == 'AKSI_ON' else '20'
                        print(f"PERINTAH: PERANGKAT 2 -> {current_action_state}")
                        send_command(cmd)
                    
                    final_command_sent = True 
                    last_valid_time = current_time_for_logic 
                    # --- [SOLUSI UTAMA] Set cooldown lebih lama setelah perintah sukses ---
                    current_cooldown_limit = POST_COMMAND_COOLDOWN 
                    print(f"SISTEM ISTIRAHAT {POST_COMMAND_COOLDOWN} DETIK (Silakan reset tangan Anda)")
                    # -------------------------------------------------------------------
                
                elif temp_action in ACTION_GESTURES:
                     pass # Abaikan

            # 2. JIKA MENUNGGU GESTUR AKSI (State Belum Aktif)
            else:
                if temp_action in ACTION_GESTURES:
                    if temp_action == 'close_to_open_palm':
                        current_action_state = 'AKSI_ON'
                    elif temp_action == 'open_to_close_palm':
                        current_action_state = 'AKSI_OFF'
                    
                    print(f"STATE SET: {current_action_state}")
                    last_action_time = current_time_for_logic
                    last_valid_time = current_time_for_logic
                    current_cooldown_limit = COOLDOWN_DURATION # Gunakan cooldown standar

        # Reset State logic
        if current_action_state and (current_time_for_logic - last_action_time > STATE_TIMEOUT):
            print("TIMEOUT: State dibatalkan.")
            current_action_state = None
        
        if final_command_sent:
            current_action_state = None

        # --- TAMPILAN VISUAL ---
        if in_cooldown:
            # Hitung mundur visual
            remaining = current_cooldown_limit - time_since_last
            color = (0, 255, 255) if current_cooldown_limit > 2.0 else (0, 165, 255) # Kuning jika long cooldown
            status_msg = "RESET TANGAN SEKARANG!" if remaining > 2.0 else "JEDA..."
            
            cv2.putText(image, f"{status_msg} ({remaining:.1f}s)", (15, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "SIAP", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        state_text = f'STATE: {current_action_state}' if current_action_state else 'STATE: Menunggu'
        cv2.putText(image, state_text, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'GESTUR: {temp_action}', (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('WIFI Control (Smart Cooldown)', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Selesai.")