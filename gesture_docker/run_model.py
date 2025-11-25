import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
import time
import requests
import socket
from requests.exceptions import ConnectionError
from zeroconf import ServiceBrowser, Zeroconf

# ==========================================
# --- KONFIGURASI SISTEM (RASPBERRY PI) ---
# ==========================================

# Konfigurasi Waktu
COOLDOWN_DURATION = 1.5         
POST_COMMAND_COOLDOWN = 3.0     
STATE_TIMEOUT = 5 
PREDICTION_THRESHOLD = 0.97

# IP Default (Hanya placeholder, akan diisi Auto-Discovery)
ESP1_IP = "0.0.0.0" 
ESP2_IP = "0.0.0.0" 

print("🚀 [INFO] Memulai Sistem Gesture Control (Auto-Discovery Mode)...")

# ==========================================
# --- BAGIAN AUTO-DISCOVERY (ZEROCONF) ---
# ==========================================
class DeviceListener:
    def __init__(self):
        self.devices = {}

    def remove_service(self, zeroconf, type, name):
        pass

    def update_service(self, zeroconf, type, name):
        pass

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            try:
                # Decode properties
                props = {k.decode('utf-8'): v.decode('utf-8') for k, v in info.properties.items()}
                
                # Filter hanya perangkat kita 'gesture-iot'
                if props.get('type') == 'gesture-iot':
                    address = socket.inet_ntoa(info.addresses[0])
                    dev_id = props.get('id')
                    print(f"   -> 🔍 Ditemukan: Perangkat {dev_id} di {address}")
                    self.devices[dev_id] = address
            except Exception as e:
                print(f"   [Warning] Gagal parse service: {e}")

def find_esp_devices():
    print("\n📡 Memindai jaringan mencari ESP8266 (5 detik)...")
    zeroconf = Zeroconf()
    listener = DeviceListener()
    browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)
    
    time.sleep(5) # Waktu tunggu scanning
    zeroconf.close()
    return listener.devices

# --- EKSEKUSI PENCARIAN DI AWAL ---
found_devices = find_esp_devices()

if '1' in found_devices:
    ESP1_IP = found_devices['1']
    print(f"✅ ESP 1 Terhubung: {ESP1_IP}")
else:
    print("⚠️  ESP 1 TIDAK DITEMUKAN (Cek power/koneksi)")

if '2' in found_devices:
    ESP2_IP = found_devices['2']
    print(f"✅ ESP 2 Terhubung: {ESP2_IP}")
else:
    print("⚠️  ESP 2 TIDAK DITEMUKAN (Cek power/koneksi)")

print("-" * 40)

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
    if target_ip == "0.0.0.0":
        print(f"❌ [ERROR] IP Target tidak diketahui. Perintah '{command}' batal.")
        return

    url = f"http://{target_ip}/{command}" 
    try:
        response = requests.get(url, timeout=2.0) 
        if response.status_code == 200:
            print(f"🚀 [BERHASIL] Mengirim ke {target_ip}: Perintah {command}")
        else:
            print(f"⚠️ [GAGAL] Status Code: {response.status_code}")
    except (ConnectionError, requests.exceptions.Timeout):
        print(f"❌ [ERROR] Gagal koneksi ke ESP di {target_ip}") 

# --- DEFINISI GESTUR ---
# (Sesuaikan urutan ini SAMA PERSIS dengan saat training)
actions = np.array([
    'close_to_open_palm', 'open_to_close_palm', 
    'close_to_one', 'open_to_one', 
    'close_to_two', 'open_to_two', 
    'close_to_three', 'open_to_three', 
    'close_to_four', 'open_to_four'
])

ACTION_GESTURES = ['close_to_open_palm', 'open_to_close_palm']
SELECTION_GESTURES = [
    'close_to_one', 'open_to_one', 
    'close_to_two', 'open_to_two', 
    'close_to_three', 'open_to_three', 
    'close_to_four', 'open_to_four'
]

sequence = []
current_action_state = None 
last_action_time = 0

# Timer Cooldown
last_valid_time = 0 
current_cooldown_limit = COOLDOWN_DURATION

# --- BUKA KAMERA ---
cap = cv2.VideoCapture(0)
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

            # Gambar landmark (Opsional, matikan jika ingin performa maksimal)
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
            # LOGIKA STATE MACHINE (DYNAMIC IP + COOLDOWN)
            # =====================================================
            
            current_time_for_logic = time.time()
            final_command_sent = False 
            
            time_since_last = current_time_for_logic - last_valid_time
            in_cooldown = time_since_last < current_cooldown_limit

            if temp_action != '...' and not in_cooldown:
                
                # 1. JIKA MENUNGGU SELEKSI PERANGKAT (State Aktif)
                if current_action_state is not None:
                    if temp_action in SELECTION_GESTURES:
                        
                        # --- ROUTING DINAMIS ---
                        target_esp_ip = "0.0.0.0"
                        device_name = ""
                        cmd = ""

                        if temp_action in ['close_to_one', 'open_to_one']:
                            target_esp_ip = ESP1_IP
                            device_name = "PERANGKAT 1"
                            cmd = '11' if current_action_state == 'AKSI_ON' else '10'
                        
                        elif temp_action in ['close_to_two', 'open_to_two']:
                            target_esp_ip = ESP2_IP
                            device_name = "PERANGKAT 2"
                            cmd = '21' if current_action_state == 'AKSI_ON' else '20'
                        
                        # Tambahkan logika untuk Device 3 & 4 jika nanti ada
                        
                        if target_esp_ip != "0.0.0.0":
                            print(f"📡 PERINTAH: {device_name} (IP: {target_esp_ip}) -> {current_action_state}")
                            send_command(cmd, target_esp_ip)
                            
                            final_command_sent = True 
                            last_valid_time = current_time_for_logic 
                            current_cooldown_limit = POST_COMMAND_COOLDOWN 
                            print(f"⏳ SISTEM ISTIRAHAT {POST_COMMAND_COOLDOWN} DETIK...")
                        else:
                            print(f"⚠️ [WARNING] {device_name} tidak ditemukan saat scanning awal.")
                    
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

            # --- TAMPILAN VISUAL ---
            # (Hanya akan muncul jika Anda menjalankan container dengan akses display/xhost)
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