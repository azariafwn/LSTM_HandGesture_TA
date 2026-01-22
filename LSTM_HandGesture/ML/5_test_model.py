import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
import time

# --- KONFIGURASI UTAMA ---
USE_VIDEO_FILE = False          # Ubah ke True untuk testing pake video
VIDEO_PATH = 'video_testing.mp4' 
COOLDOWN_DURATION = 2.0        # Jeda waktu (detik) setelah deteksi berhasil
PREDICTION_THRESHOLD = 0.97    # Ambang batas kepercayaan
# -------------------------

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
TFLITE_MODEL_PATH = 'C:/zafaa/kuliah/SEMESTER7/PRATA/code_gesture/LSTM_HandGesture/model.tflite'
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model TFLite berhasil dimuat.")

# --- DEFINISI GESTUR (8 KELAS) ---
actions = np.array(['close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three', 'close_to_four', 'open_to_four'])

sequence = []
current_action = '...'
last_valid_time = 0 # Variabel untuk mencatat waktu terakhir deteksi

# --- INISIALISASI INPUT ---
if USE_VIDEO_FILE:
    print(f"MODE: Menggunakan Video File ({VIDEO_PATH})")
    cap = cv2.VideoCapture(VIDEO_PATH)
else:
    print("MODE: Menggunakan Kamera Live")
    cap = cv2.VideoCapture(0) # Ganti ke 1 jika pakai webcam eksternal
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    print("Error: Tidak bisa membuka sumber input.")
    exit()

prev_time = 0
fps = 0

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"RESOLUSI KAMERA SAAT INI: {w} x {h}")
# ---------------------

print("\nMulai deteksi... Tekan 'q' untuk keluar.")

with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
    while cap.isOpened():
        curr_time = time.time()
        if prev_time > 0: fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        ret, frame = cap.read()
        
        # Logika Loop Video
        if not ret:
            if USE_VIDEO_FILE:
                print("Video selesai. Mengulang...")
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
        confidence = 0.0

        # --- LOGIKA DETEKSI ---
        if len(sequence) == 30:
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            predicted_class_index = np.argmax(prediction)
            confidence = prediction[0][predicted_class_index]

            if confidence > PREDICTION_THRESHOLD:
                temp_action = actions[predicted_class_index]

            # Visualisasi Probabilitas
            y_pos = 60
            for i, action_name in enumerate(actions):
                if i > 3: break # Tampilkan 4 teratas saja agar layar tidak penuh
                prob_text = f"{action_name}: {prediction[0][i]:.2f}"
                color = (0, 255, 0) if i == predicted_class_index and confidence > PREDICTION_THRESHOLD else (0, 0, 0)
                cv2.putText(image, prob_text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                y_pos += 20

        # --- LOGIKA COOLDOWN ---
        time_since_last = curr_time - last_valid_time
        in_cooldown = time_since_last < COOLDOWN_DURATION

        if temp_action != '...':
            if not in_cooldown:
                # Jika gestur valid DAN tidak sedang cooldown -> PROSES
                print(f"GESTUR TERDETEKSI: {temp_action} ({confidence*100:.1f}%)")
                current_action = temp_action
                last_valid_time = curr_time # Reset timer cooldown
            else:
                # Jika sedang cooldown, abaikan (atau beri log debug)
                # print(f"Mengabaikan {temp_action} (Cooldown: {COOLDOWN_DURATION - time_since_last:.1f}s)")
                pass
        
        # --- TAMPILAN VISUAL ---
        # Status Cooldown di Layar
        if in_cooldown:
            cv2.putText(image, f"JEDA: {COOLDOWN_DURATION - time_since_last:.1f}s", (image.shape[1] - 150, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "SIAP DETEKSI", (image.shape[1] - 180, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(image, f'FPS: {int(fps)}', (image.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'LAST ACTION: {current_action}', (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('OpenCV Feed - Test Model (Cooldown)', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Selesai.")