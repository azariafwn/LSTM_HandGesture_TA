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

# --- KONFIGURASI UTAMA ---
USE_VIDEO_FILE = False
VIDEO_PATH = 'video_testing.mp4'
COOLDOWN_DURATION = 1.5         
POST_COMMAND_COOLDOWN = 4.0     

ESP1_IP = "0.0.0.0" 
ESP2_IP = "0.0.0.0" 
ESP3_IP = "0.0.0.0"
ESP4_IP = "0.0.0.0"

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
        try:
            info = zeroconf.get_service_info(type, name)
            if info:
                props = {}
                for k, v in info.properties.items():
                    try:
                        key_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                        if v is not None:
                            val_str = v.decode('utf-8') if isinstance(v, bytes) else str(v)
                        else:
                            val_str = ""
                        props[key_str] = val_str
                    except Exception:
                        continue 
                
                if props.get('type') == 'gesture-iot':
                    address = socket.inet_ntoa(info.addresses[0])
                    dev_id = props.get('id')
                    print(f"   -> Ditemukan: Perangkat {dev_id} di {address}")
                    self.devices[dev_id] = address
        except Exception as e:
            print(f"   [Info] Skip perangkat asing: {name} ({e})")

def find_esp_devices():
    print("\n📡 Memindai jaringan mencari ESP8266 (5 detik)...")
    zeroconf = Zeroconf()
    listener = DeviceListener()
    browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)
    
    time.sleep(5) 
    zeroconf.close()
    return listener.devices

# --- EKSEKUSI PENCARIAN DI AWAL ---
found_devices = find_esp_devices()

# --- Cek IP untuk 4 Perangkat ---
if '1' in found_devices:
    ESP1_IP = found_devices['1']
    print(f"✅ ESP 1 Terhubung: {ESP1_IP}")
else:
    print("⚠️  ESP 1 TIDAK DITEMUKAN")

if '2' in found_devices:
    ESP2_IP = found_devices['2']
    print(f"✅ ESP 2 Terhubung: {ESP2_IP}")
else:
    print("⚠️  ESP 2 TIDAK DITEMUKAN")

if '3' in found_devices:
    ESP3_IP = found_devices['3']
    print(f"✅ ESP 3 Terhubung: {ESP3_IP}")
else:
    print("⚠️  ESP 3 TIDAK DITEMUKAN")

if '4' in found_devices:
    ESP4_IP = found_devices['4']
    print(f"✅ ESP 4 Terhubung: {ESP4_IP}")
else:
    print("⚠️  ESP 4 TIDAK DITEMUKAN")

print("-" * 40)
# ==========================================

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

def send_command(command, target_ip):
    if target_ip == "0.0.0.0":
        print(f"❌ Error: IP Target belum terdeteksi. Tidak bisa kirim perintah {command}.")
        return

    url = f"http://{target_ip}/{command}" 
    try:
        response = requests.get(url, timeout=2.0)
        if response.status_code == 200:
            print(f"BERHASIL MENGIRIM ke {target_ip}: Perintah {command}")
        else:
            print(f"Gagal mengirim ke {target_ip}: Status {response.status_code}")
    except (ConnectionError, requests.exceptions.Timeout):
        print(f"Peringatan: Gagal koneksi ke ESP di {target_ip} ({command})") 

# --- Update Daftar Gestur ---
# Pastikan urutan ini SAMA PERSIS dengan urutan saat training model!
actions = np.array([
    'close_to_open_palm', 'open_to_close_palm', 
    'close_to_one', 'open_to_one', 
    'close_to_two', 'open_to_two', 
    'close_to_three', 'open_to_three', 
    'close_to_four', 'open_to_four'
])

ACTION_GESTURES = ['close_to_open_palm', 'open_to_close_palm']
# Tambahkan gestur seleksi untuk 3 dan 4
SELECTION_GESTURES = [
    'close_to_one', 'open_to_one', 
    'close_to_two', 'open_to_two', 
    'close_to_three', 'open_to_three', 
    'close_to_four', 'open_to_four'
]

sequence = []
current_action_state = None 
last_action_time = 0
STATE_TIMEOUT = 5 

last_valid_time = 0 
current_cooldown_limit = COOLDOWN_DURATION
PREDICTION_THRESHOLD = 0.97

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
        # LOGIKA STATE MACHINE 4 DEVICES
        # =====================================================
        
        current_time_for_logic = time.time()
        final_command_sent = False 
        
        time_since_last = current_time_for_logic - last_valid_time
        in_cooldown = time_since_last < current_cooldown_limit

        if temp_action != '...' and not in_cooldown:
            
            if current_action_state is not None:
                if temp_action in SELECTION_GESTURES:
                    
                    # --- ROUTING KE 4 DEVICES ---
                    target_esp_ip = "0.0.0.0"
                    device_name = ""
                    cmd = ""

                    # --- Device 1 ---
                    if temp_action in ['close_to_one', 'open_to_one']:
                        target_esp_ip = ESP1_IP
                        device_name = "PERANGKAT 1"
                        cmd = '11' if current_action_state == 'AKSI_ON' else '10'
                    
                    # --- Device 2 ---
                    elif temp_action in ['close_to_two', 'open_to_two']:
                        target_esp_ip = ESP2_IP
                        device_name = "PERANGKAT 2"
                        cmd = '21' if current_action_state == 'AKSI_ON' else '20'
                    
                    # --- Device 3 ---
                    elif temp_action in ['close_to_three', 'open_to_three']:
                        target_esp_ip = ESP3_IP
                        device_name = "PERANGKAT 3"
                        cmd = '31' if current_action_state == 'AKSI_ON' else '30'

                    # --- Device 4 ---
                    elif temp_action in ['close_to_four', 'open_to_four']:
                        target_esp_ip = ESP4_IP
                        device_name = "PERANGKAT 4"
                        cmd = '41' if current_action_state == 'AKSI_ON' else '40'
                    
                    # Eksekusi
                    if target_esp_ip != "0.0.0.0":
                        print(f"PERINTAH: {device_name} (IP: {target_esp_ip}) -> {current_action_state}")
                        send_command(cmd, target_esp_ip)
                        
                        final_command_sent = True 
                        last_valid_time = current_time_for_logic 
                        current_cooldown_limit = POST_COMMAND_COOLDOWN 
                        print(f"SISTEM ISTIRAHAT {POST_COMMAND_COOLDOWN} DETIK (Silakan reset tangan Anda)")
                    else:
                        print(f"❌ Error: {device_name} tidak ditemukan saat scanning awal.")

                elif temp_action in ACTION_GESTURES:
                    pass 

            else:
                if temp_action in ACTION_GESTURES:
                    if temp_action == 'close_to_open_palm':
                        current_action_state = 'AKSI_ON'
                    elif temp_action == 'open_to_close_palm':
                        current_action_state = 'AKSI_OFF'
                    
                    print(f"STATE SET: {current_action_state}")
                    last_action_time = current_time_for_logic
                    last_valid_time = current_time_for_logic
                    current_cooldown_limit = COOLDOWN_DURATION 

        # Reset State logic
        if current_action_state and (current_time_for_logic - last_action_time > STATE_TIMEOUT):
            print("TIMEOUT: State dibatalkan.")
            current_action_state = None
        
        if final_command_sent:
            current_action_state = None

        # --- TAMPILAN VISUAL ---
        if in_cooldown:
            remaining = current_cooldown_limit - time_since_last
            color = (0, 255, 255) if current_cooldown_limit > 2.0 else (0, 165, 255)
            status_msg = "RESET TANGAN SEKARANG!" if remaining > 2.0 else "JEDA..."
            cv2.putText(image, f"{status_msg} ({remaining:.1f}s)", (15, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "SIAP", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        state_text = f'STATE: {current_action_state}' if current_action_state else 'STATE: Menunggu'
        cv2.putText(image, state_text, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'GESTUR: {temp_action}', (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('WIFI Control (4 Devices)', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Selesai.")