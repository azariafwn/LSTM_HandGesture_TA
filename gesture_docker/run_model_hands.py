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
OUTPUT_DIR = "logs_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOG_FILE = os.path.join(OUTPUT_DIR, "data_pengujian.csv")

# --- Variabel Global ---
CURRENT_RESOLUTION_STR = "?"
SELECTED_DISTANCE_STR = "Unknown"
TARGET_GESTURE_STR = "Unknown" # Placeholder untuk target gestur yang akan dipilih
# -----------------------------------------------------

# Buat header CSV jika file belum ada (UPDATE HEADER)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        # --- Tambah Header "Target_Gesture" ---
        # Urutan: ..., Resolution, Distance, Target (Seharusnya), Last Command (Realisasi)
        writer.writerow(["Timestamp", "Event", "FPS", "Edge_Latency_ms", "WiFi_Latency_ms", "Total_Latency_ms", "Resolution", "Distance", "Target_Gesture", "Last_Command"])
        # ---------------------------------------------------------------

# --- Update fungsi log ---
# Menerima parameter 'target_gesture' tambahan
def log_to_csv(event, fps, edge_ms, wifi_ms, total_ms, resolution, distance, target_gesture, last_cmd_str):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # Masukkan data target_gesture ke baris CSV
        writer.writerow([now, event, f"{fps:.2f}", f"{edge_ms:.2f}", f"{wifi_ms:.2f}", f"{total_ms:.2f}", resolution, distance, target_gesture, last_cmd_str])
# ---------------------------------------------

# ==========================================
# --- KONFIGURASI SISTEM ---
# ==========================================
COOLDOWN_DURATION = 1.5
POST_COMMAND_COOLDOWN = 3.5
STATE_TIMEOUT = 5
PREDICTION_THRESHOLD = 0.97

ESP1_IP = "0.0.0.0"
ESP2_IP = "0.0.0.0"
ESP3_IP = "0.0.0.0"
ESP4_IP = "0.0.0.0"

print("🚀 [INFO] Memulai Sistem Gesture Control...")

# --- BAGIAN AUTO-DISCOVERY ---
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

# ========================================================
# --- MEDIAPIPE SETUP ---
# ========================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    return rh

# --- LOAD MODEL TFLITE ---
TFLITE_MODEL_PATH = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- FUNGSI KIRIM ---
def send_command(command, target_ip):
    if target_ip == "0.0.0.0": return 0

    url = f"http://{target_ip}/{command}"
    try:
        start_net = time.time()
        response = requests.get(url, timeout=2.0)
        end_net = time.time()

        latency_ms = (end_net - start_net) * 1000
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

# ==========================================
# --- SETUP KAMERA & RESOLUSI ---
# ==========================================
cap = cv2.VideoCapture(0)
# Atur resolusi tinggi untuk layar seleksi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# --- Tentukan String Resolusi ---
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if h == 480: CURRENT_RESOLUTION_STR = "480p"
elif h == 720: CURRENT_RESOLUTION_STR = "720p"
elif h == 1080: CURRENT_RESOLUTION_STR = "1080p"
else: CURRENT_RESOLUTION_STR = f"{h}p"

print(f"\n[INFO KAMERA] Resolusi Aktif: {CURRENT_RESOLUTION_STR}")
time.sleep(1)

# ==============================================================================
# --- MENU 1: LOOP SELEKSI JARAK ---
# ==============================================================================
print("--- MEMULAI MENU 1: SELEKSI JARAK ---")
selecting_distance = True
while selecting_distance and cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    # Judul Menu 1
    cv2.putText(frame, "MENU 1: PILIH JARAK PENGUJIAN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(frame, "MENU 1: PILIH JARAK PENGUJIAN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Pilihan
    cv2.putText(frame, "[1] Jarak DEKAT (30cm)", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "[2] Jarak SEDANG (50cm)",(80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "[3] Jarak JAUH (70cm)",  (80, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.imshow('Raspi Gesture Control', frame)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('1'): SELECTED_DISTANCE_STR = "30cm"; selecting_distance = False
    elif key == ord('2'): SELECTED_DISTANCE_STR = "50cm"; selecting_distance = False
    elif key == ord('3'): SELECTED_DISTANCE_STR = "70cm"; selecting_distance = False
    elif key == ord('q'): cap.release(); cv2.destroyAllWindows(); exit()

print(f"✅ JARAK TERPILIH: {SELECTED_DISTANCE_STR}")
time.sleep(0.5) # Jeda antar menu

# ==============================================================================
# --- MENU 2: LOOP SELEKSI TARGET GESTURE ---
# ==============================================================================
print("--- MEMULAI MENU 2: SELEKSI TARGET GESTURE ---")
selecting_target = True
while selecting_target and cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    # Judul Menu 2
    cv2.putText(frame, "MENU 2: PILIH TARGET GESTURE (YG AKAN DIUJI)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(frame, "MENU 2: PILIH TARGET GESTURE (YG AKAN DIUJI)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"Jarak terpilih: {SELECTED_DISTANCE_STR}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Pilihan Kolom Kiri (ON)
    cv2.putText(frame, "[1] D1 ON",  (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "[3] D2 ON",  (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "[5] D3 ON",  (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.putText(frame, "[7] D4 ON",  (50, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Pilihan Kolom Kanan (OFF)
    cv2.putText(frame, "[2] D1 OFF", (300, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.putText(frame, "[4] D2 OFF", (300, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
    cv2.putText(frame, "[6] D3 OFF", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    cv2.putText(frame, "[8] D4 OFF", (300, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 255), 2)

    cv2.imshow('Raspi Gesture Control', frame)
    key = cv2.waitKey(10) & 0xFF

    # Logika Pemilihan Target (Angka 1-8)
    if key == ord('1'): TARGET_GESTURE_STR = "D1 ON"; selecting_target = False
    elif key == ord('2'): TARGET_GESTURE_STR = "D1 OFF"; selecting_target = False
    elif key == ord('3'): TARGET_GESTURE_STR = "D2 ON"; selecting_target = False
    elif key == ord('4'): TARGET_GESTURE_STR = "D2 OFF"; selecting_target = False
    elif key == ord('5'): TARGET_GESTURE_STR = "D3 ON"; selecting_target = False
    elif key == ord('6'): TARGET_GESTURE_STR = "D3 OFF"; selecting_target = False
    elif key == ord('7'): TARGET_GESTURE_STR = "D4 ON"; selecting_target = False
    elif key == ord('8'): TARGET_GESTURE_STR = "D4 OFF"; selecting_target = False
    elif key == ord('q'): cap.release(); cv2.destroyAllWindows(); exit()

print(f"✅ TARGET TERPILIH: {TARGET_GESTURE_STR}")
print("\nMemulai deteksi gestur dalam 2 detik...")
time.sleep(2)
# ==============================================================================


# ==========================================
# --- MAIN LOOP GESTURE DETECTION ---
# ==========================================

prev_time = 0
fps = 0

# --- INISIALISASI MP HANDS ---
with mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        start_edge = time.time()

        ret, frame = cap.read()
        if not ret: break

        # Deteksi Tangan
        image, results = mediapipe_detection(frame, hands)

        # Gambar Landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Ekstrak Keypoints
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

        end_edge = time.time()
        edge_latency_ms = (end_edge - start_edge) * 1000

        # ==========================================
        # LOGIKA STATE MACHINE & PENGIRIMAN
        # ==========================================
        current_time_for_logic = time.time()
        final_command_sent = False
        wifi_latency_ms = 0

        time_since_last = current_time_for_logic - last_valid_time
        in_cooldown = time_since_last < current_cooldown_limit

        if temp_action != '...' and not in_cooldown:
            if current_action_state is not None:
                if temp_action in SELECTION_GESTURES:
                    target_esp_ip = "0.0.0.0"
                    cmd = ""
                    target_device_id = ""

                    # --- Set target_device_id ---
                    if 'one' in temp_action:
                        target_esp_ip = ESP1_IP; cmd = '11' if 'open' in temp_action else '10'
                        target_device_id = "D1"
                    elif 'two' in temp_action:
                        target_esp_ip = ESP2_IP; cmd = '21' if 'open' in temp_action else '20'
                        target_device_id = "D2"
                    elif 'three' in temp_action:
                        target_esp_ip = ESP3_IP; cmd = '31' if 'open' in temp_action else '30'
                        target_device_id = "D3"
                    elif 'four' in temp_action:
                        target_esp_ip = ESP4_IP; cmd = '41' if 'open' in temp_action else '40'
                        target_device_id = "D4"
                    # -------------------------------------------------

                    real_cmd = cmd[0] + ('1' if current_action_state == 'AKSI_ON' else '0')

                    if target_esp_ip != "0.0.0.0":
                        now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        print(f"[{now}] 🎯 Gestur {temp_action} terdeteksi. Mengirim ke {target_esp_ip} ({target_device_id})...")

                        # Kirim Perintah
                        wifi_latency_ms = send_command(real_cmd, target_esp_ip)
                        total_latency = edge_latency_ms + wifi_latency_ms

                        # --- Buat string perintah terakhir (REALISASI) ---
                        status_str = "ON" if current_action_state == 'AKSI_ON' else "OFF"
                        last_command_str = f"{target_device_id} {status_str}"

                        # --- Log ke CSV termasuk TARGET GESTURE ---
                        # Parameter: ..., resolution, distance, target_gesture, last_command
                        log_to_csv("COMMAND_SENT", fps, edge_latency_ms, wifi_latency_ms, total_latency, CURRENT_RESOLUTION_STR, SELECTED_DISTANCE_STR, TARGET_GESTURE_STR, last_command_str)
                        # ----------------------------------------------------------

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

        # --- TAMPILAN VISUAL UTAMA ---
        if in_cooldown:
            remaining = current_cooldown_limit - time_since_last
            status_msg = "RESET TANGAN!" if remaining > 2.0 else "JEDA..."
            color = (0, 255, 255) if remaining > 2.0 else (0, 165, 255)
            cv2.putText(image, f"{status_msg} ({remaining:.1f}s)", (15, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "SIAP (HANDS MODE)", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        state_text = f'STATE: {current_action_state}' if current_action_state else 'STATE: Menunggu'
        cv2.putText(image, state_text, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # --- Info Pojok Kanan Atas ---
        # Resolusi
        cv2.putText(image, f'Res: {CURRENT_RESOLUTION_STR}', (image.shape[1] - 220, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        # Jarak
        cv2.putText(image, f'Dist: {SELECTED_DISTANCE_STR}', (image.shape[1] - 220, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2, cv2.LINE_AA)
        # --- Tampilkan Target di Layar Utama ---
        cv2.putText(image, f'Target: {TARGET_GESTURE_STR}', (image.shape[1] - 220, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        # ------------------------------------------------------------

        cv2.putText(image, f'FPS: {int(fps)}', (image.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Raspi Gesture Control', image)
        if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()