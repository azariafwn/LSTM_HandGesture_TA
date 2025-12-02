import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
import time
import requests
import socket
import csv
import datetime
from requests.exceptions import ConnectionError
from zeroconf import ServiceBrowser, Zeroconf

# ==========================================
# --- KONFIGURASI PENGUKURAN DATA ---
# ==========================================
LOG_FILE = "data_skripsi.csv"

# Buat header CSV jika file belum ada
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Event", "FPS", "Edge_Latency_ms", "WiFi_Latency_ms", "Total_Latency_ms"])

def log_to_csv(event, fps, edge_ms, wifi_ms, total_ms):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        writer.writerow([now, event, f"{fps:.2f}", f"{edge_ms:.2f}", f"{wifi_ms:.2f}", f"{total_ms:.2f}"])

# ==========================================
# --- KONFIGURASI SISTEM ---
# ==========================================
COOLDOWN_DURATION = 1.5         
POST_COMMAND_COOLDOWN = 3.0     
STATE_TIMEOUT = 5 
PREDICTION_THRESHOLD = 0.97

ESP1_IP = "0.0.0.0" 
ESP2_IP = "0.0.0.0" 
ESP3_IP = "0.0.0.0" 
ESP4_IP = "0.0.0.0" 

print("🚀 [INFO] Memulai Sistem Gesture Control + DATA LOGGING...")

# --- BAGIAN AUTO-DISCOVERY (SAMA SEPERTI SEBELUMNYA) ---
class DeviceListener:
    def __init__(self):
        self.devices = {}
    def remove_service(self, zeroconf, type, name): pass
    def update_service(self, zeroconf, type, name): pass
    def add_service(self, zeroconf, type, name):
        try:
            info = zeroconf.get_service_info(type, name)
            if info:
                props = {}
                for k, v in info.properties.items():
                    try:
                        key_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                        val_str = v.decode('utf-8') if isinstance(v, bytes) else str(v) if v else ""
                        props[key_str] = val_str
                    except: continue 
                if props.get('type') == 'gesture-iot':
                    address = socket.inet_ntoa(info.addresses[0])
                    self.devices[props.get('id')] = address
        except: pass

def find_esp_devices():
    print("\n📡 Memindai jaringan mencari ESP8266 (5 detik)...")
    zeroconf = Zeroconf()
    listener = DeviceListener()
    browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)
    time.sleep(5) 
    zeroconf.close()
    return listener.devices

found_devices = find_esp_devices()
if '1' in found_devices: ESP1_IP = found_devices['1']; print(f"✅ ESP 1: {ESP1_IP}")
if '2' in found_devices: ESP2_IP = found_devices['2']; print(f"✅ ESP 2: {ESP2_IP}")
if '3' in found_devices: ESP3_IP = found_devices['3']; print(f"✅ ESP 3: {ESP3_IP}")
if '4' in found_devices: ESP4_IP = found_devices['4']; print(f"✅ ESP 4: {ESP4_IP}")
print("-" * 40)

# --- INISIALISASI MEDIAPIPE & MODEL ---
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

TFLITE_MODEL_PATH = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- FUNGSI KIRIM DENGAN PENGUKURAN LATENSI ---
def send_command(command, target_ip):
    if target_ip == "0.0.0.0": return 0 # Return 0 ms jika gagal

    url = f"http://{target_ip}/{command}" 
    try:
        # UKUR LATENSI WIFI
        start_net = time.time()
        response = requests.get(url, timeout=2.0) 
        end_net = time.time()
        
        latency_ms = (end_net - start_net) * 1000
        
        # Format Timestamp untuk Laporan (Bab 4.6)
        now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if response.status_code == 200:
            print(f"[{now}] 📡 HTTP OK dari {target_ip} (Latensi: {latency_ms:.1f}ms)")
            return latency_ms
        else:
            print(f"[{now}] ⚠️ HTTP FAIL {response.status_code}")
            return 0
    except:
        print(f"❌ Error Koneksi")
        return 0

# --- VARIABEL ---
actions = np.array(['close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three', 'close_to_four', 'open_to_four'])
SELECTION_GESTURES = ['close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three', 'close_to_four', 'open_to_four']
ACTION_GESTURES = ['close_to_open_palm', 'open_to_close_palm']

sequence = []
current_action_state = None 
last_action_time = 0
last_valid_time = 0 
current_cooldown_limit = COOLDOWN_DURATION

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

prev_time = 0
fps = 0

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        # --- UKUR FPS ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        # Mulai ukur Edge Latency (Processing Time)
        start_edge = time.time()

        ret, frame = cap.read()
        if not ret: break

        image, results = mediapipe_detection(frame, holistic)
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

        # Hitung waktu selesai pemrosesan AI
        end_edge = time.time()
        edge_latency_ms = (end_edge - start_edge) * 1000

        # Logika State Machine
        current_time_for_logic = time.time()
        final_command_sent = False 
        wifi_latency_ms = 0 # Default 0

        time_since_last = current_time_for_logic - last_valid_time
        in_cooldown = time_since_last < current_cooldown_limit

        if temp_action != '...' and not in_cooldown:
            if current_action_state is not None:
                if temp_action in SELECTION_GESTURES:
                    target_esp_ip = "0.0.0.0"
                    cmd = ""
                    
                    # Logic Mapping (Sederhana untuk logging)
                    if 'one' in temp_action: target_esp_ip = ESP1_IP; cmd = '11' if 'open' in temp_action else '10' # Contoh simple logic
                    elif 'two' in temp_action: target_esp_ip = ESP2_IP; cmd = '21' if 'open' in temp_action else '20'
                    elif 'three' in temp_action: target_esp_ip = ESP3_IP; cmd = '31' if 'open' in temp_action else '30'
                    elif 'four' in temp_action: target_esp_ip = ESP4_IP; cmd = '41' if 'open' in temp_action else '40'
                    
                    # Perbaikan logic ON/OFF sesuai state
                    real_cmd = cmd[0] + ('1' if current_action_state == 'AKSI_ON' else '0')

                    if target_esp_ip != "0.0.0.0":
                        now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"[{now}] 🎯 Gestur {temp_action} terdeteksi. Mengirim ke {target_esp_ip}...")
                        
                        # KIRIM DAN CATAT LATENSI WIFI
                        wifi_latency_ms = send_command(real_cmd, target_esp_ip)
                        
                        # LOG DATA KE CSV (Hanya saat ada aksi kirim)
                        total_latency = edge_latency_ms + wifi_latency_ms
                        log_to_csv("COMMAND_SENT", fps, edge_latency_ms, wifi_latency_ms, total_latency)

                        final_command_sent = True 
                        last_valid_time = current_time_for_logic 
                        current_cooldown_limit = POST_COMMAND_COOLDOWN 
                        print(f"⏳ Istirahat {POST_COMMAND_COOLDOWN}s...")

            else:
                if temp_action in ACTION_GESTURES:
                    if temp_action == 'close_to_open_palm': current_action_state = 'AKSI_ON'
                    elif temp_action == 'open_to_close_palm': current_action_state = 'AKSI_OFF'
                    
                    now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"[{now}] 🔄 State berubah: {current_action_state}")
                    last_action_time = current_time_for_logic
                    last_valid_time = current_time_for_logic
                    current_cooldown_limit = COOLDOWN_DURATION 

        if current_action_state and (current_time_for_logic - last_action_time > STATE_TIMEOUT):
            print("❌ TIMEOUT")
            current_action_state = None
        if final_command_sent: current_action_state = None

        # Print FPS Log Rutin (untuk Screenshot 4.5)
        # Print setiap 30 frame agar tidak spam
        if int(curr_time * 10) % 30 == 0: 
            print(f"Inference time: {edge_latency_ms:.1f}ms | FPS: {fps:.1f}")

        # Tampilan (Jika ada monitor)
        cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Raspi Gesture', image)
        if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()