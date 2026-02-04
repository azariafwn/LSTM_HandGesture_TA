import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
import time
import requests
import csv
import datetime
from flask import Flask, Response, render_template_string

# ==========================================
# --- INISIALISASI FLASK ---
# ==========================================
app = Flask(__name__)

# ==========================================
# --- KONFIGURASI PATH ---
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, 'model.tflite')

OUTPUT_DIR = os.path.join(BASE_DIR, "logs_output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOG_FILE = os.path.join(OUTPUT_DIR, "data_pengujian_stream.csv")

# Setting Default
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
PREDICTION_THRESHOLD = 0.97

print("🚀 [INFO] Memulai Sistem Gesture Control (Mode Web Stream)...")

# Setup MediaPipe Global (Resource ringan, aman global)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    return rh

def send_command(command, target_ip):
    # Dummy send command untuk mode testing flask
    if target_ip == "0.0.0.0": return 0
    return 0

actions = np.array(['close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three', 'close_to_four', 'open_to_four'])
SELECTION_GESTURES = ['close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three', 'close_to_four', 'open_to_four']
ACTION_GESTURES = ['close_to_open_palm', 'open_to_close_palm']

# ==========================================
# --- FUNGSI GENERATOR VIDEO (CORE LOGIC) ---
# ==========================================
def generate_frames():
    # 1. SETUP KAMERA
    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 2. SETUP MODEL TFLITE (LOKAL - WAJIB DI SINI AGAR THREAD SAFE)
    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, num_threads=4)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ Model TFLite initialized for new stream thread.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. SETUP VARIABEL STATE
    sequence = []
    current_action_state = None
    last_action_time = 0
    last_valid_time = 0
    current_cooldown_limit = COOLDOWN_DURATION
    prev_time = 0

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0, # Ringan
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # --- PENTING: FLIP FRAME (MIRRORING) ---
            # Ini seringkali ngaruh ke kenyamanan deteksi tangan
            frame = cv2.flip(frame, 1)

            # Hitung FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time
            start_edge = time.time()

            # --- PROSES AI ---
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw Landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Prediction Logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            temp_action = '...'
            
            if len(sequence) == 30:
                input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
                # Gunakan interpreter LOKAL yang dibuat di atas
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])

                if np.max(prediction) > PREDICTION_THRESHOLD:
                    temp_action = actions[np.argmax(prediction)]

            end_edge = time.time()
            edge_latency_ms = (end_edge - start_edge) * 1000

            # --- STATE MACHINE (SIMPLIFIED) ---
            current_time_for_logic = time.time()
            time_since_last = current_time_for_logic - last_valid_time
            in_cooldown = time_since_last < current_cooldown_limit
            final_command_sent = False

            if temp_action != '...' and not in_cooldown:
                if current_action_state is not None:
                    if temp_action in SELECTION_GESTURES:
                        # Simulasi Log
                        status_str = "ON" if current_action_state == 'AKSI_ON' else "OFF"
                        log_to_csv("COMMAND_SENT", fps, edge_latency_ms, 0, edge_latency_ms, CURRENT_RESOLUTION_STR, SELECTED_DISTANCE_STR, temp_action, f"CMD {status_str}", SELECTED_LIGHT_LUX)
                        
                        final_command_sent = True
                        last_valid_time = current_time_for_logic
                        current_cooldown_limit = POST_COMMAND_COOLDOWN
                else:
                    if temp_action in ACTION_GESTURES:
                        if temp_action == 'close_to_open_palm': current_action_state = 'AKSI_ON'
                        elif temp_action == 'open_to_close_palm': current_action_state = 'AKSI_OFF'
                        last_action_time = current_time_for_logic
                        last_valid_time = current_time_for_logic
                        current_cooldown_limit = COOLDOWN_DURATION

            if current_action_state and (current_time_for_logic - last_action_time > STATE_TIMEOUT):
                current_action_state = None
            if final_command_sent: current_action_state = None

            # --- DRAWING ---
            status_color = (0, 255, 0)
            if in_cooldown:
                status_color = (0, 255, 255)
                cv2.putText(image, "COOLDOWN", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            cv2.putText(image, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            state_text = f'STATE: {current_action_state}' if current_action_state else 'STATE: Menunggu'
            cv2.putText(image, state_text, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # --- ENCODE TO JPEG ---
            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    print("❌ Camera released.")

# ==========================================
# --- ROUTES FLASK ---
# ==========================================
@app.route('/')
def index():
    return render_template_string('''
        <html>
        <head>
            <title>TA SmartHome Monitor</title>
            <style>
                body { background-color: #121212; color: #eee; font-family: sans-serif; text-align: center; }
                img { border: 2px solid #0f0; max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Live Monitor</h1>
            <img src="{{ url_for('video_feed') }}">
        </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# --- MAIN ---
# ==========================================
if __name__ == "__main__":
    # threaded=True SANGAT PENTING agar stream tidak memblokir server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)