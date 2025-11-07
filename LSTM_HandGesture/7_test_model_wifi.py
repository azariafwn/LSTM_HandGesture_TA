import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
import time
import requests # <-- Menggantikan 'serial'
from requests.exceptions import ConnectionError # <-- Untuk menangani error

# --- PENGATURAN KONEKSI WI-FI KE ESP8266 ---
ESP_IP = "10.141.159.103" # <-- GANTI INI DENGAN IP ANDA
BASE_URL = f"http://{ESP_IP}"

print(f"Mencoba terhubung ke server ESP8266 di {BASE_URL}...")
# ---------------------------------------------

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

# --- PENGATURAN MODEL TFLITE ---
TFLITE_MODEL_PATH = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model TFLite berhasil dimuat.")
# ----------------------------------

# --- FUNGSI BARU UNTUK MENGIRIM PERINTAH VIA WI-FI ---
def send_command(command):
    """Mengirim perintah ke ESP8266 dan menangani error koneksi."""
    try:
        # Timeout 0.5 detik agar tidak memblokir loop utama
        response = requests.get(f"{BASE_URL}/{command}", timeout=0.5)
        # Cek apakah request berhasil (status code 200)
        if response.status_code == 200:
            print(f"BERHASIL MENGIRIM: Perintah {command}")
        else:
            print(f"Gagal mengirim perintah {command}: Status {response.status_code}")
    except ConnectionError:
        print(f"Error: Tidak bisa terhubung ke {BASE_URL}. Periksa koneksi/IP.")
    except requests.exceptions.Timeout:
        print(f"Error: Request ke {BASE_URL}/{command} timeout.")
# ----------------------------------------------------

# --- LOGIKA STATE BARU ---
# Variabel untuk menyimpan state
current_action_state = None # Akan berisi 'AKSI_ON' atau 'AKSI_OFF'
last_action_time = 0
STATE_TIMEOUT = 5 # Waktu (detik) untuk memilih perangkat sebelum state di-reset

# Variabel untuk debounce gestur
last_detected_gesture = '...' 

# --- Definisikan gestur untuk logika yang lebih bersih ---
ACTION_GESTURES = ['close_to_open_palm', 'open_to_close_palm']
SELECTION_GESTURES = ['close_to_one', 'close_to_two', 'open_to_one', 'open_to_two']
# -----------------------------------------------------------------

actions = np.array(['close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'close_to_two', 'open_to_one', 'open_to_two'])
sequence = []

prediction_threshold = 0.97 # threshold disesuaikan sama akurasi model

cap = cv2.VideoCapture(0)

# --- PENINGKATAN FPS ---
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# ------------------------

prev_time = 0
fps = 0

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        curr_time = time.time()
        if prev_time > 0:
            fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        ret, frame = cap.read()
        if not ret: break

        image, results = mediapipe_detection(frame, holistic)

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                results.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Reset aksi sementara jika tidak ada tangan terdeteksi
        temp_action = '...'

        if len(sequence) == 30:
            input_data = np.expand_dims(sequence, axis=0)
            input_data = np.array(input_data, dtype=np.float32)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])

            predicted_class_index = np.argmax(prediction)
            confidence = prediction[0][predicted_class_index]

            if confidence > prediction_threshold:
                temp_action = actions[predicted_class_index]

            # Tampilkan probabilitas
            prob_text_1 = f"{actions[0]}: {prediction[0][0]:.2f} | {actions[1]}: {prediction[0][1]:.2f}"
            prob_text_2 = f"{actions[2]}: {prediction[0][2]:.2f} | {actions[3]}: {prediction[0][3]:.2f}"

            cv2.putText(image, prob_text_1, (15, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, prob_text_2, (15, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # --- [MODIFIKASI UTAMA] Logika STATE MACHINE ---
        
        current_time_for_logic = time.time()
        final_command_sent = False # Flag untuk reset state

        # 1. Cek apakah gestur baru terdeteksi (Debounce)
        if temp_action != '...' and temp_action != last_detected_gesture:
            print(f"GESTUR BARU TERDETEKSI: {temp_action}")
            last_detected_gesture = temp_action # Update debounce

            # --- 2. Logika State v3 (Lebih Tangguh) ---
            
            # A. PERIKSA JIKA STATE SEDANG AKTIF (Menunggu Seleksi Perangkat)
            if current_action_state is not None:
                
                if temp_action in SELECTION_GESTURES:
                    # Perangkat 1
                    if temp_action == 'close_to_one' or temp_action == 'open_to_one':
                        if current_action_state == 'AKSI_ON':
                            print("PERINTAH FINAL: NYALAKAN PERANGKAT 1")
                            send_command('11')
                        elif current_action_state == 'AKSI_OFF':
                            print("PERINTAH FINAL: MATIKAN PERANGKAT 1")
                            send_command('10')
                    
                    # PERANGKAT_2
                    elif temp_action == 'close_to_two' or temp_action == 'open_to_two':
                        if current_action_state == 'AKSI_ON':
                            print("PERINTAH FINAL: NYALAKAN PERANGKAT 2")
                            send_command('21')
                        elif current_action_state == 'AKSI_OFF':
                            print("PERINTAH FINAL: MATIKAN PERANGKAT 2")
                            send_command('20')
                    
                    final_command_sent = True # Tandai untuk reset state

                elif temp_action in ACTION_GESTURES:
                    # INI SOLUSINYA: Pengguna melakukan gestur Aksi (lagi)
                    # Kita ABAIKAN gestur ini dan biarkan state tetap berjalan.
                    print(f"LOGIKA: Mengabaikan gestur aksi '{temp_action}' karena menunggu gestur seleksi.")
                    pass 

            # B. JIKA STATE TIDAK AKTIF (Menunggu Gestur Aksi)
            else:
                if temp_action in ACTION_GESTURES:
                    # Ini yang kita mau! Pengguna memulai aksi.
                    if temp_action == 'close_to_open_palm':
                        current_action_state = 'AKSI_ON'
                    elif temp_action == 'open_to_close_palm':
                        current_action_state = 'AKSI_OFF'
                    
                    last_action_time = current_time_for_logic
                    print(f"STATE: {current_action_state}. Pilih perangkat (dalam {STATE_TIMEOUT} dtk)...")

                elif temp_action in SELECTION_GESTURES:
                    # Pengguna memilih perangkat tanpa memilih aksi dulu.
                    print(f"LOGIKA: Gestur '{temp_action}' diabaikan. Lakukan gestur aksi (ON/OFF) dulu.")
                    pass

        # 3. Cek Timeout & Reset State
        if current_action_state and (current_time_for_logic - last_action_time > STATE_TIMEOUT):
            print(f"STATE: Timeout. Aksi '{current_action_state}' dibatalkan.")
            current_action_state = None
            last_action_time = 0
        
        if final_command_sent:
            print("STATE: Perintah terkirim. State di-reset.")
            current_action_state = None
            last_action_time = 0
        # ----------------------------------------------------

        cv2.putText(image, f'FPS: {int(fps)}', (image.shape[1] - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'FRAMES: {len(sequence)}/30', (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f'GESTUR: {last_detected_gesture}', (15, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        state_text = f'STATE: {current_action_state}' if current_action_state else 'STATE: Menunggu Aksi'
        cv2.putText(image, state_text, (15, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed - Tes Model TFLite (WIFI)', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
# Tidak perlu 'ser.close()' lagi
print("Selesai.")