import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
import time
import requests
import socket
import csv
import sqlite3 # <--- TAMBAHAN BARU
import datetime
from requests.exceptions import ConnectionError
from flask import Flask, Response, render_template_string
from zeroconf import ServiceBrowser, Zeroconf

# ==========================================
# --- INISIALISASI FLASK ---
# ==========================================
app = Flask(__name__)

# ==========================================
# --- KONFIGURASI PATH & LOGGING ---
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, 'model.tflite')

OUTPUT_DIR = os.path.join(BASE_DIR, "logs_output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOG_FILE = os.path.join(OUTPUT_DIR, "data_pengujian_raspi.csv")
DB_FILE = os.path.join(OUTPUT_DIR, "logs_raspi.db") # <--- DATABASE FILE

# Setting Default
CURRENT_RESOLUTION_STR = "480p"
SELECTED_DISTANCE_STR = "50cm"
SELECTED_LIGHT_LUX = 150
TARGET_GESTURE_STR = "Realtime"

# --- INIT CSV ---
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Event", "FPS", "Edge_Latency_ms", "WiFi_Latency_ms", "Total_Latency_ms", "Resolution", "Distance", "Target_Gesture", "Last_Command", "Light_Intensity_Lux"])

# --- INIT DATABASE (SQLITE) ---
def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Buat tabel jika belum ada
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                fps REAL,
                edge_latency_ms REAL,
                wifi_latency_ms REAL,
                total_latency_ms REAL,
                resolution TEXT,
                distance TEXT,
                target_gesture TEXT,
                last_command TEXT,
                light_intensity_lux INTEGER
            )
        ''')
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {DB_FILE}")
    except Exception as e:
        print(f"❌ Database Init Error: {e}")

# Panggil fungsi init saat startup
init_db()

def log_to_data(event, fps, edge_ms, wifi_ms, total_ms, resolution, distance, target_gesture, last_cmd_str, light_lux):
    now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    # 1. Simpan ke CSV (Backup Legacy)
    try:
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([now, event, f"{fps:.2f}", f"{edge_ms:.2f}", f"{wifi_ms:.2f}", f"{total_ms:.2f}", resolution, distance, target_gesture, last_cmd_str, light_lux])
    except Exception as e:
        print(f"CSV Error: {e}")

    # 2. Simpan ke SQLite (Untuk Web Laravel)
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO activity_logs (timestamp, event, fps, edge_latency_ms, wifi_latency_ms, total_latency_ms, resolution, distance, target_gesture, last_command, light_intensity_lux)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (now, event, fps, edge_ms, wifi_ms, total_ms, resolution, distance, target_gesture, last_cmd_str, light_lux))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

# ==========================================
# --- KONFIGURASI SISTEM ---
# ==========================================
COOLDOWN_DURATION = 1.5
POST_COMMAND_COOLDOWN = 2.5
STATE_TIMEOUT = 5
PREDICTION_THRESHOLD = 0.97

# --- VARIABEL IP ESP (Akan diisi otomatis oleh Zeroconf) ---
ESP1_IP = "0.0.0.0" 
ESP2_IP = "0.0.0.0"
ESP3_IP = "0.0.0.0"
ESP4_IP = "0.0.0.0"

print("🚀 [INFO] Memulai Sistem Gesture Control (Flask + SQLite)...")

# ========================================================
# --- AUTO DISCOVERY (ZEROCONF) ---
# ========================================================
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
# --- MEDIAPIPE & TFLITE SETUP ---
# ========================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"✅ Model loaded: {TFLITE_MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

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

def send_command(command, target_ip):
    if target_ip == "0.0.0.0": return 0
    
    url = f"http://{target_ip}/{command}"
    try:
        start_net = time.time()
        response = requests.get(url, timeout=1.0) 
        end_net = time.time()
        latency_ms = (end_net - start_net) * 1000
        
        if response.status_code == 200:
            print(f"📡 HTTP 200 OK -> {target_ip}")
            return latency_ms
        else:
            print(f"⚠️ HTTP FAIL {response.status_code}")
            return 0
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return 0

actions = np.array(['close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three', 'close_to_four', 'open_to_four'])
SELECTION_GESTURES = ['close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three', 'close_to_four', 'open_to_four']
ACTION_GESTURES = ['close_to_open_palm', 'open_to_close_palm']

# ==========================================
# --- CORE LOGIC GENERATOR ---
# ==========================================
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    sequence = []
    current_action_state = None
    last_action_time = 0
    last_valid_time = 0
    current_cooldown_limit = COOLDOWN_DURATION
    prev_time = 0
    fps = 0

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while True:
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            start_edge = time.time()
            image, results = mediapipe_detection(frame, hands)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            temp_action = '...'
            max_prob = 0.0

            if len(sequence) == 30:
                input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])

                max_prob = np.max(prediction)
                predicted_class = actions[np.argmax(prediction)]

                if max_prob > PREDICTION_THRESHOLD:
                    temp_action = predicted_class

            end_edge = time.time()
            edge_latency_ms = (end_edge - start_edge) * 1000

            # --- LOGIKA STATE MACHINE ---
            current_time_for_logic = time.time()
            final_command_sent = False
            wifi_latency_ms = 0
            time_since_last = current_time_for_logic - last_valid_time
            in_cooldown = time_since_last < current_cooldown_limit

            if temp_action != '...' and not in_cooldown:
                if current_action_state is not None:
                    # LOGIKA SELEKSI DEVICE
                    if temp_action in SELECTION_GESTURES:
                        target_device_id = ""
                        target_ip = "0.0.0.0"
                        cmd_prefix = ""
                        
                        if 'one' in temp_action: target_ip = ESP1_IP; target_device_id="D1"; cmd_prefix="1"
                        elif 'two' in temp_action: target_ip = ESP2_IP; target_device_id="D2"; cmd_prefix="2"
                        elif 'three' in temp_action: target_ip = ESP3_IP; target_device_id="D3"; cmd_prefix="3"
                        elif 'four' in temp_action: target_ip = ESP4_IP; target_device_id="D4"; cmd_prefix="4"

                        status_suffix = "1" if current_action_state == 'AKSI_ON' else "0"
                        real_cmd = cmd_prefix + status_suffix
                        
                        # --- KIRIM PERINTAH NYATA KE LAMPU ---
                        if target_ip != "0.0.0.0":
                            print(f"🚀 MENGIRIM REQUEST ke {target_ip}...")
                            wifi_latency_ms = send_command(real_cmd, target_ip)
                        else:
                            print("⚠️ ESP IP belum ditemukan!")
                        
                        status_str = "ON" if current_action_state == 'AKSI_ON' else "OFF"
                        # GANTI FUNGSI LOG KE YANG BARU (DB + CSV)
                        log_to_data("COMMAND_SENT", fps, edge_latency_ms, wifi_latency_ms, edge_latency_ms + wifi_latency_ms, CURRENT_RESOLUTION_STR, SELECTED_DISTANCE_STR, temp_action, f"{target_device_id} {status_str}", SELECTED_LIGHT_LUX)
                        
                        final_command_sent = True
                        last_valid_time = current_time_for_logic
                        current_cooldown_limit = POST_COMMAND_COOLDOWN

                else:
                    if temp_action in ACTION_GESTURES:
                        if temp_action == 'close_to_open_palm': current_action_state = 'AKSI_ON'
                        elif temp_action == 'open_to_close_palm': current_action_state = 'AKSI_OFF'
                        
                        print(f"🔄 STATE CHANGED TO: {current_action_state}")
                        last_action_time = current_time_for_logic
                        last_valid_time = current_time_for_logic
                        current_cooldown_limit = COOLDOWN_DURATION

            if current_action_state and (current_time_for_logic - last_action_time > STATE_TIMEOUT):
                current_action_state = None
            if final_command_sent: current_action_state = None

            # Overlay
            debug_color = (0, 255, 255) if max_prob > PREDICTION_THRESHOLD else (0, 0, 255)
            raw_text = f"Raw: {predicted_class if len(sequence)==30 else '...'} ({max_prob:.2f})"
            cv2.putText(image, raw_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, debug_color, 2)

            if in_cooldown:
                cv2.putText(image, "COOLDOWN", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(image, "SIAP", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            state_text = f'{current_action_state}' if current_action_state else 'Menunggu State...'
            state_color = (0, 255, 0) if current_action_state == 'AKSI_ON' else ((0, 0, 255) if current_action_state == 'AKSI_OFF' else (200, 200, 200))
            cv2.putText(image, state_text, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head>
            <title>Raspi Gesture Monitor</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { background-color: #1a1a1a; color: #fff; font-family: sans-serif; text-align: center; margin: 0; padding: 20px; }
                img { width: 100%; max-width: 640px; border: 2px solid #444; }
                .info { margin-top: 10px; color: #aaa; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <h2>Raspberry Pi Live Monitor</h2>
            <img src="{{ url_for('video_feed') }}">
            <div class="info">
                <p>Status: Flask Streaming Active + SQLite Logging</p>
                <p>Ready for Laravel Dashboard Integration</p>
            </div>
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)