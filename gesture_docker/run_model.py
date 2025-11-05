# Nama file: 7_test_model_raspi.py
import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
import time
import lgpio  # <-- Menggantikan 'serial'

# ==================================
# --- PENGATURAN GPIO RASPBERRY PI ---
# ==================================
# Gunakan nomor pin BCM (GPIO), bukan nomor pin fisik.
RELAY_PIN_1 = 17  # Ganti jika perlu (misal: GPIO 17)
RELAY_PIN_2 = 27  # Ganti jika perlu (misal: GPIO 27)

# Di Raspberry Pi 5, 40-pin header terhubung ke chip 4
GPIO_CHIP = 4 

# Tentukan logika relay Anda
# 1 = HIGH = ON
# 0 = LOW  = OFF
RELAY_ON = 1
RELAY_OFF = 0

def setup_gpio():
    """Membuka chip GPIO Pi 5, mengklaim 2 pin, dan mengembalikannya."""
    try:
        # Buka chip GPIO utama Pi 5
        h = lgpio.gpiochip_open(GPIO_CHIP)
        
        # Klaim pin 1 sebagai output
        lgpio.gpio_claim_output(h, RELAY_PIN_1)
        # Klaim pin 2 sebagai output
        lgpio.gpio_claim_output(h, RELAY_PIN_2)
        
        # Pastikan kedua relay OFF saat program dimulai
        lgpio.gpio_write(h, RELAY_PIN_1, RELAY_OFF)
        lgpio.gpio_write(h, RELAY_PIN_2, RELAY_OFF)
        
        print(f"✅ [INFO] GPIO chip {GPIO_CHIP} dibuka.")
        print(f"✅ [INFO] Pin {RELAY_PIN_1} (Relay 1) & {RELAY_PIN_2} (Relay 2) siap.")
        return h
    except Exception as e:
        print(f"❌ GAGAL setup GPIO: {e}")
        print("Pastikan Docker container dijalankan dengan flag '--device=/dev/gpiochip4'")
        print("Dan/atau user di host Pi ada di grup 'gpio'.")
        return None

def cleanup_gpio(h):
    """Mematikan relay dan menutup handle GPIO."""
    if h:
        print("\n[INFO] Membersihkan GPIO...")
        lgpio.gpio_write(h, RELAY_PIN_1, RELAY_OFF) # Matikan relay 1
        lgpio.gpio_write(h, RELAY_PIN_2, RELAY_OFF) # Matikan relay 2
        lgpio.gpiochip_close(h)                     # Tutup handle
        print("[INFO] GPIO cleanup selesai.")

# ==================================
# --- KODE DETEKSI (Sama seperti file 6) ---
# ==================================

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
print("✅ [INFO] Model TFLite berhasil dimuat.")
# ----------------------------------

actions = np.array(['thumbs_down_to_up', 'thumbs_up_to_down', 'close_to_open_palm', 'open_to_close_palm'])
sequence = []

current_action = '...'
last_sent_action = '...' 

# Sesuaikan threshold jika perlu
prediction_threshold = 0.97 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
print("✅ [INFO] Kamera webcam berhasil dibuka.")

prev_time = 0
fps = 0

# ==================================
# --- MAIN LOOP DENGAN GPIO ---
# ==================================

gpio_handle = setup_gpio()
if not gpio_handle:
    cap.release()
    cv2.destroyAllWindows()
    exit()

try:
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

                prob_text_1 = f"{actions[0]}: {prediction[0][0]:.2f} | {actions[1]}: {prediction[0][1]:.2f}"
                prob_text_2 = f"{actions[2]}: {prediction[0][2]:.2f} | {actions[3]}: {prediction[0][3]:.2f}"
                cv2.putText(image, prob_text_1, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, prob_text_2, (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # --- Logika pengiriman sinyal GPIO (PENGGANTI SERIAL) ---
            if temp_action != last_sent_action:
                
                # Logika dari kode ESP32 Anda:
                # '1' -> Relay 1 ON
                # '0' -> Relay 1 OFF
                # '2' -> Relay 2 ON
                # '3' -> Relay 2 OFF
                
                if temp_action == 'thumbs_down_to_up':
                    lgpio.gpio_write(gpio_handle, RELAY_PIN_1, RELAY_ON)
                    print("GPIO ACTION: Relay 1 ON 💡")
                elif temp_action == 'thumbs_up_to_down':
                    lgpio.gpio_write(gpio_handle, RELAY_PIN_1, RELAY_OFF)
                    print("GPIO ACTION: Relay 1 OFF 🔌")
                
                elif temp_action == 'close_to_open_palm':
                    lgpio.gpio_write(gpio_handle, RELAY_PIN_2, RELAY_ON)
                    print("GPIO ACTION: Relay 2 ON 💡")
                elif temp_action == 'open_to_close_palm':
                    lgpio.gpio_write(gpio_handle, RELAY_PIN_2, RELAY_OFF)
                    print("GPIO ACTION: Relay 2 OFF 🔌")

                last_sent_action = temp_action
            # ----------------------------------------------------

            current_action = last_sent_action
            
            cv2.putText(image, f'FPS: {int(fps)}', (image.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'FRAMES: {len(sequence)}/30', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'GESTUR: {current_action}', (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Tampilkan feed (memerlukan X11 forwarding dari Docker)
            cv2.imshow('Raspberry Pi - TFLite Gesture Control', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

finally:
    # Blok ini akan selalu berjalan, bahkan jika ada error di 'try'
    cap.release()
    cv2.destroyAllWindows()
    cleanup_gpio(gpio_handle) # Pastikan relay mati
    print("Program ditutup, semua sumber daya dibebaskan.")