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
from flask import Flask, Response, render_template_string

# ==========================================
# --- INISIALISASI FLASK ---
# ==========================================
app = Flask(__name__)

# ==========================================
# --- KONFIGURASI PATH & LOGGING ---
# ==========================================
# Mundur satu level folder untuk akses model & logs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, 'model.tflite')

OUTPUT_DIR = os.path.join(BASE_DIR, "logs_output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOG_FILE = os.path.join(OUTPUT_DIR, "data_pengujian_stream.csv")

# Setting Default (Hardcoded)
CURRENT_RESOLUTION_STR = "480p"
SELECTED_DISTANCE_STR = "50cm"
SELECTED_LIGHT_LUX = 150
TARGET_GESTURE_STR = "Realtime"

# Buat Header CSV
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Event", "FPS", "Edge_Latency_ms", "WiFi_Latency_ms", "Total_Latency_ms", "Resolution", "Distance", "Target_Gesture", "Last_Command", "Light_Intensity_Lux"])

def log_to_csv(event, fps, edge_ms, wifi_ms, total_ms, resolution, distance, target_gesture, last_cmd_str, light_lux):
    try:
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            writer.writerow([now, event, f"{fps:.2f}", f"{edge_ms:.2f}", f"{wifi_ms:.2f}", f"{total_ms:.2f}", resolution, distance, target_gesture, last_cmd_str, light_lux])
    except Exception as e:
        print(f"Error logging: {e}")

# ==========================================
# --- KONFIGURASI SISTEM ---
# ==========================================
COOLDOWN_DURATION = 1.5
POST_COMMAND_COOLDOWN = 2.5
STATE_TIMEOUT = 5
PREDICTION_THRESHOLD = 0.97 # Ambang batas deteksi

ESP1_IP = "0.0.0.0" # Default dummy
ESP2_IP = "0.0.0.0"
ESP3_IP = "0.0.0.0"
ESP4_IP = "0.0.0.0"

# --- NETWORK DISCOVERY (Dummy untuk Test Flask) ---
# Kita skip zeroconf yang kompleks biar gak blocking di Windows
print("🚀 [INFO] Memulai Sistem Gesture Control (Mode Web Stream)...")


# ========================================================
# --- MEDIAPIPE & TFLITE SETUP ---
# ========================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load Model
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
    # Dummy send command untuk testing di Windows
    if target_ip == "0.0.0.0": return 0
    return 0

# --- VARIABEL STATE GLOBAL ---
actions = np.array(['close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three', 'close_to_four', 'open_to_four'])
SELECTION_GESTURES = ['close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three', 'close_to_four', 'open_to_four']
ACTION_GESTURES = ['close_to_open_palm', 'open_to_close_palm']

# ==========================================
# --- CORE LOGIC GENERATOR ---
# ==========================================
def generate_frames():
    # Setup Kamera (0 = Laptop Default)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Variabel Loop Utama (Persis run_model_hands.py)
    sequence = []
    current_action_state = None
    last_action_time = 0
    last_valid_time = 0
    current_cooldown_limit = COOLDOWN_DURATION
    prev_time = 0
    fps = 0

    # Inisialisasi MediaPipe
    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while True:
            # 1. Baca Frame
            success, frame = cap.read()
            if not success:
                break

            # Hitung FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            start_edge = time.time()

            # 2. Deteksi MediaPipe (Persis Logika Asli)
            image, results = mediapipe_detection(frame, hands)

            # 3. Gambar Landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 4. Ekstrak Keypoints & Prediksi TFLite
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:] # Keep last 30 frames

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

            # 5. Logika State Machine & Cooldown (Persis Logika Asli)
            current_time_for_logic = time.time()
            final_command_sent = False
            wifi_latency_ms = 0
            time_since_last = current_time_for_logic - last_valid_time
            in_cooldown = time_since_last < current_cooldown_limit

            if temp_action != '...' and not in_cooldown:
                if current_action_state is not None:
                    if temp_action in SELECTION_GESTURES:
                        # Logika dummy IP mapping agar tidak error
                        target_device_id = "TEST_DEV"
                        # Simulasi log command sent
                        status_str = "ON" if current_action_state == 'AKSI_ON' else "OFF"
                        
                        # Log ke CSV
                        log_to_csv("COMMAND_SENT", fps, edge_latency_ms, wifi_latency_ms, edge_latency_ms, CURRENT_RESOLUTION_STR, SELECTED_DISTANCE_STR, temp_action, f"CMD {status_str}", SELECTED_LIGHT_LUX)
                        
                        print(f"🎯 GESTURE: {temp_action} -> CMD: {status_str}")
                        
                        final_command_sent = True
                        last_valid_time = current_time_for_logic
                        current_cooldown_limit = POST_COMMAND_COOLDOWN

                else:
                    if temp_action in ACTION_GESTURES:
                        if temp_action == 'close_to_open_palm': current_action_state = 'AKSI_ON'
                        elif temp_action == 'open_to_close_palm': current_action_state = 'AKSI_OFF'
                        
                        print(f"🔄 STATE CHANGE: {current_action_state}")
                        last_action_time = current_time_for_logic
                        last_valid_time = current_time_for_logic
                        current_cooldown_limit = COOLDOWN_DURATION

            if current_action_state and (current_time_for_logic - last_action_time > STATE_TIMEOUT):
                current_action_state = None
            if final_command_sent: current_action_state = None

            # 6. Overlay Tampilan (Persis Tampilan Asli)
            if in_cooldown:
                remaining = current_cooldown_limit - time_since_last
                status_msg = "RESET TANGAN!" if remaining > 2.0 else "JEDA..."
                color = (0, 255, 255) if remaining > 2.0 else (0, 165, 255)
                cv2.putText(image, f"{status_msg} ({remaining:.1f}s)", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "SIAP (HANDS MODE)", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            state_text = f'STATE: {current_action_state}' if current_action_state else 'STATE: Menunggu'
            cv2.putText(image, state_text, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Info Pojok Kanan Atas
            cv2.putText(image, f'Res: {CURRENT_RESOLUTION_STR}', (image.shape[1] - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(image, f'Dist: {SELECTED_DISTANCE_STR}', (image.shape[1] - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
            cv2.putText(image, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 7. ENCODE KE JPEG (KUNCI STREAMING)
            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()

            # Yield frame ke browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# ==========================================
# --- ROUTES FLASK ---
# ==========================================
@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head>
            <title>SmartHome Gesture Monitor</title>
            <style>
                body { background-color: #1a1a1a; color: #fff; font-family: sans-serif; text-align: center; }
                h1 { margin-top: 20px; color: #00ff00; }
                .video-container { margin: 20px auto; border: 3px solid #333; display: inline-block; }
                img { width: 100%; max-width: 800px; height: auto; }
            </style>
        </head>
        <body>
            <h1>Live Monitor - Local Test</h1>
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}">
            </div>
            <p>Status: Running on Flask (Windows Local)</p>
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# --- MAIN ENTRY ---
# ==========================================
if __name__ == "__main__":
    # Jalankan Flask
    # 'threaded=True' penting agar request web tidak memblokir loop kamera
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)