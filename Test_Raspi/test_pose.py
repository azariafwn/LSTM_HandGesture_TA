import cv2
import time
import lgpio  # <-- Menggantikan 'serial'
import math
import mediapipe as mp

# =========================
# KONFIGURASI
# =========================
# Gunakan nomor pin BCM (GPIO), bukan nomor pin fisik.
# Pin fisik 11 adalah GPIO 17.
RELAY_PIN = 17

SEND_COOLDOWN = 1.0  # detik; cegah spam perintah

# =========================
# GPIO HANDLER (menggunakan lgpio)
# =========================
def setup_gpio():
    """Membuka chip GPIO, mengklaim pin sebagai output, dan mengembalikan handle."""
    try:
        # Buka chip GPIO utama (biasanya chip 0)
        h = lgpio.gpiochip_open(0)
        # Klaim pin sebagai output
        lgpio.gpio_claim_output(h, RELAY_PIN)
        print(f"✅ [INFO] GPIO pin {RELAY_PIN} berhasil disiapkan sebagai output.")
        return h
    except Exception as e:
        print(f"❌ GAGAL setup GPIO: {e}")
        print("Pastikan program dijalankan dengan izin yang cukup (sudo jika perlu).")
        return None

# =========================
# GESTURE DETECTOR (Tidak ada perubahan di sini, sama seperti kodemu)
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def is_finger_folded(hand, tip_idx, pip_idx):
    return hand.landmark[tip_idx].y > hand.landmark[pip_idx].y

def classify_gesture(hand):
    lm = hand.landmark
    thumb_tip = lm[4]
    thumb_mcp = lm[2]
    index_finger_pip = lm[6]
    index_finger_mcp = lm[5]

    folded_index = is_finger_folded(hand, 8, 6)
    folded_middle = is_finger_folded(hand, 12, 10)
    folded_ring = is_finger_folded(hand, 16, 14)
    folded_pinky = is_finger_folded(hand, 20, 18)
    all_others_folded = folded_index and folded_middle and folded_ring and folded_pinky

    if not all_others_folded:
        return "NONE"

    thumb_points_up = thumb_tip.y < thumb_mcp.y
    thumb_is_above_fist = thumb_tip.y < index_finger_pip.y
    if thumb_points_up and thumb_is_above_fist:
        return "THUMBS_UP"

    thumb_points_down = thumb_tip.y > thumb_mcp.y
    thumb_is_below_fist = thumb_tip.y > index_finger_mcp.y
    if thumb_points_down and thumb_is_below_fist:
        return "THUMBS_DOWN"

    return "NONE"

def main():
    gpio_handle = setup_gpio()
    # Jika setup GPIO gagal, program tidak akan lanjut
    if not gpio_handle:
        return

    # Atur relay ke kondisi OFF (0) di awal program
    lgpio.gpio_write(gpio_handle, RELAY_PIN, 0)
    print("[INFO] Relay diatur ke OFF pada awal program.")

    # Buka kamera lokal (webcam yang terhubung ke USB Raspi)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ [ERROR] Gagal membuka kamera. Pastikan webcam terhubung.")
        lgpio.gpiochip_close(gpio_handle) # Selalu cleanup
        return

    print("✅ [INFO] Berhasil terhubung ke kamera webcam.")

    last_sent = "OFF"
    last_send_time = 0
    prev_time = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            current_time = time.time()
            # Hindari pembagian dengan nol saat pertama kali dijalankan
            if (current_time - prev_time) > 0:
                fps = 1 / (current_time - prev_time)
            else:
                fps = 0
            prev_time = current_time

            ok, frame = cap.read()
            if not ok:
                print("[WARN] Gagal membaca frame dari kamera.")
                continue

            # Flip frame agar seperti cermin, lebih intuitif
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            gesture = "NONE"

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                gesture = classify_gesture(hand)
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                x_coords = [landmark.x for landmark in hand.landmark]
                y_coords = [landmark.y for landmark in hand.landmark]
                x_min = int(min(x_coords) * w) - 25
                y_min = int(min(y_coords) * h) - 25
                x_max = int(max(x_coords) * w) + 25
                y_max = int(max(y_coords) * h) + 25
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, gesture, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            now = time.time()
            # Logika untuk mengirim perintah ke GPIO
            if gesture == "THUMBS_UP" and (last_sent != "ON") and (now - last_send_time > SEND_COOLDOWN):
                lgpio.gpio_write(gpio_handle, RELAY_PIN, 1) # <-- Set GPIO ke HIGH (1)
                print("[ACTION] Relay ON 💡")
                last_sent = "ON"
                last_send_time = now

            elif gesture == "THUMBS_DOWN" and (last_sent != "OFF") and (now - last_send_time > SEND_COOLDOWN):
                lgpio.gpio_write(gpio_handle, RELAY_PIN, 0) # <-- Set GPIO ke LOW (0)
                print("[ACTION] Relay OFF 🔌")
                last_sent = "OFF"
                last_send_time = now

            # Tampilkan status dan FPS di layar
            status_text = f"Last Sent: {last_sent or '-'}"
            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {int(fps)}", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Tampilkan jendela di VNC
            cv2.imshow("Gesture -> Relay (tekan 'q' untuk keluar)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    print("Menutup program...")
    cap.release()
    cv2.destroyAllWindows()
    # Cleanup GPIO
    lgpio.gpio_write(gpio_handle, RELAY_PIN, 0) # Pastikan relay mati
    lgpio.gpiochip_close(gpio_handle) # Lepaskan handle GPIO

if _name_ == "_main_":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna (Ctrl+C).")