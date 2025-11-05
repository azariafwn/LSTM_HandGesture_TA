import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
import time

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

actions = np.array(['thumbs_down_to_up', 'thumbs_up_to_down', 'close_to_open_palm', 'open_to_close_palm'])
sequence = []

# --- Variabel untuk debounce/mencegah spam print ke terminal ---
current_action = '...'
last_sent_action = '...' 

# Anda bisa turunkan ini ke 0.90 agar sesuai dengan akurasi 93.75% model terakhir Anda
prediction_threshold = 0.95 

cap = cv2.VideoCapture(0)

# --- PENINGKATAN FPS ---
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
# ------------------------

if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

prev_time = 0
fps = 0

print("\nMulai deteksi... Tekan 'q' untuk keluar.")

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

                # Tampilan probabilitas (visual)
                prob_text_1 = f"{actions[0]}: {prediction[0][0]:.2f} | {actions[1]}: {prediction[0][1]:.2f}"
                prob_text_2 = f"{actions[2]}: {prediction[0][2]:.2f} | {actions[3]}: {prediction[0][3]:.2f}"

                cv2.putText(image, prob_text_1, (15, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, prob_text_2, (15, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # --- Cetak ke Terminal ---
            # Hanya cetak jika ada perubahan aksi (mencegah spam)
            if temp_action != last_sent_action:
                
                # Hanya cetak jika aksinya bukan "diam"
                if temp_action != '...':
                    print(f"GESTUR BARU TERDETEKSI: {temp_action}")

                last_sent_action = temp_action

            current_action = last_sent_action # Update tampilan di layar
            # ----------------------------------------------------

            cv2.putText(image, f'FPS: {int(fps)}', (image.shape[1] - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'FRAMES: {len(sequence)}/30', (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'GESTUR: {current_action}', (15, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed - Tes Model TFLite', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nMenutup program...")
finally:
    # Selalu pastikan kamera dan jendela ditutup
    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")