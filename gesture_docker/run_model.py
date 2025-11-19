import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
import time
import requests
from requests.exceptions import ConnectionError

# ==========================================
# --- KONFIGURASI SISTEM (RASPBERRY PI) ---
# ==========================================
# IP Address ESP8266 (Pastikan RPi satu jaringan dengan ESP)
ESP1_IP = "10.141.159.103"  # <-- IP ESP 1 (Device 1)
ESP2_IP = "10.141.159.149"  # <-- IP ESP 2 (Device 2)

# Konfigurasi Waktu
COOLDOWN_DURATION = 1.5         
POST_COMMAND_COOLDOWN = 4.0     
STATE_TIMEOUT = 5 
PREDICTION_THRESHOLD = 0.97

print(f"✅ [INFO] Konfigurasi: Device 1 @ {ESP1_IP}, Device 2 @ {ESP2_IP}")

# --- INISIALISASI MEDIAPIPE ---
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

# --- LOAD MODEL TFLITE ---
TFLITE_MODEL_PATH = 'model.tflite'
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ [INFO] Model TFLite berhasil dimuat.")
except Exception as e:
    print(f"❌ [ERROR] Gagal memuat model: {e}")
    exit()

# --- FUNGSI KIRIM PERINTAH WI-FI ---
def send_command(command, target_ip):
    url = f"http://{target_ip}/{command}" 
    try:
        # Timeout diset 2 detik
        response = requests.get(url, timeout=2.0) 
        if response.status_code == 200:
            print(f"🚀 [BERHASIL] Mengirim ke {target_ip}: Perintah {command}")
        else:
            print(f"⚠️ [GAGAL] Status Code: {response.status_code}")
    except (ConnectionError, requests.exceptions.Timeout):
        print(f"❌ [ERROR] Gagal koneksi ke ESP di {target_ip}") 

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

# Timer Cooldown
last_valid_time = 0 
current_cooldown_limit = COOLDOWN_DURATION

# --- BUKA KAMERA ---
cap = cv2.VideoCapture(0)
# Atur resolusi rendah agar performa RPi lebih cepat (opsional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    print("❌ [ERROR] Kamera tidak ditemukan!")
    exit()

print("✅ [INFO] Kamera dibuka. Mulai deteksi...")

prev_time = 0
fps = 0

try:
    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while cap.isOpened():
            curr_time = time.time()
            if prev_time > 0: fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            ret, frame = cap.read()
            if not ret: break

            image, results = mediapipe_detection(frame, holistic)

            # Gambar landmark (Opsional, matikan jika ingin lebih cepat)
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
            # LOGIKA STATE MACHINE (DUAL ESP + COOLDOWN)
            # =====================================================
            
            current_time_for_logic = time.time()
            final_command_sent = False 
            
            time_since_last = current_time_for_logic - last_valid_time
            in_cooldown = time_since_last < current_cooldown_limit

            if temp_action != '...' and not in_cooldown:
                
                # 1. JIKA MENUNGGU SELEKSI PERANGKAT (State Aktif)
                if current_action_state is not None:
                    if temp_action in SELECTION_GESTURES:
                        
                        if temp_action in ['close_to_one', 'open_to_one']:
                            # DEVICE 1 -> Kirim ke ESP1_IP
                            cmd = '11' if current_action_state == 'AKSI_ON' else '10'
                            print(f"📡 PERINTAH: PERANGKAT 1 -> {current_action_state}")
                            send_command(cmd, ESP1_IP) 
                        
                        elif temp_action in ['close_to_two', 'open_to_two']:
                            # DEVICE 2 -> Kirim ke ESP2_IP
                            cmd = '21' if current_action_state == 'AKSI_ON' else '20'
                            print(f"📡 PERINTAH: PERANGKAT 2 -> {current_action_state}")
                            send_command(cmd, ESP2_IP)
                        
                        final_command_sent = True 
                        last_valid_time = current_time_for_logic 
                        current_cooldown_limit = POST_COMMAND_COOLDOWN 
                        print(f"⏳ SISTEM ISTIRAHAT {POST_COMMAND_COOLDOWN} DETIK...")
                    
                    elif temp_action in ACTION_GESTURES:
                        pass 

                # 2. JIKA MENUNGGU GESTUR AKSI
                else:
                    if temp_action in ACTION_GESTURES:
                        if temp_action == 'close_to_open_palm':
                            current_action_state = 'AKSI_ON'
                        elif temp_action == 'open_to_close_palm':
                            current_action_state = 'AKSI_OFF'
                        
                        print(f"🔄 STATE SET: {current_action_state}")
                        last_action_time = current_time_for_logic
                        last_valid_time = current_time_for_logic
                        current_cooldown_limit = COOLDOWN_DURATION 

            # Reset State logic
            if current_action_state and (current_time_for_logic - last_action_time > STATE_TIMEOUT):
                print("❌ TIMEOUT: State dibatalkan.")
                current_action_state = None
            
            if final_command_sent:
                current_action_state = None

            # --- TAMPILAN VISUAL (PENTING UNTUK DEBUGGING) ---
            # Pastikan Anda menjalankan Docker dengan akses display jika ingin melihat ini
            if in_cooldown:
                remaining = current_cooldown_limit - time_since_last
                status_msg = "RESET TANGAN!" if remaining > 2.0 else "JEDA..."
                cv2.putText(image, f"{status_msg} ({remaining:.1f}s)", (15, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "SIAP", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            state_text = f'STATE: {current_action_state}' if current_action_state else 'STATE: Menunggu'
            cv2.putText(image, state_text, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'FPS: {int(fps)}', (image.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Raspi Gesture Control', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

except Exception as e:
    print(f"CRITICAL ERROR: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Program ditutup.")