# Deteksi Gestur Tangan (LSTM) dengan TFLite dan Docker

Ini adalah proyek untuk mendeteksi gestur tangan secara *real-time* (thumbs up/down) menggunakan model Jaringan Saraf Tiruan (Neural Network) berjenis LSTM. Model dilatih menggunakan data *keypoints* yang diekstrak oleh MediaPipe dan dideploy sebagai paket Docker yang *portable* dan independen.

Proyek ini dirancang untuk alur kerja *cross-platform*, di mana *development* dan *training* model dilakukan di PC Windows, sementara *deployment* (penjalanan model) ditujukan untuk perangkat Linux ARM64 (seperti Raspberry Pi 5) menggunakan Docker.

---

## 📁 Struktur Proyek

Repositori ini dibagi menjadi dua alur kerja utama yang masing-masing ada di foldernya sendiri:

* `LSTM_HandGesture/`
    Folder ini berisi semua skrip untuk alur kerja *data science*—mulai dari mengumpulkan data, memproses, melatih, hingga mengonversi model. Ini adalah "dapur" tempat model Anda dibuat.

* `gesture_docker/`
    Folder ini berisi semua file yang diperlukan untuk *deployment*—`Dockerfile`, skrip inferensi `run_model.py`, dan `requirements.txt` untuk membuat image Docker yang siap pakai.

---

## ⚙️ Penjelasan Alur Kode

Berikut adalah penjelasan rinci dari setiap skrip dalam proyek ini.

### 1. `LSTM_HandGesture/` (Alur Training & Konversi)

* **`1_collect_data.py`**
    * **Tujuan:** Merekam video mentah dari gestur Anda.
    * **Cara Kerja:** Menggunakan OpenCV untuk membuka webcam. Skrip akan meminta Anda merekam sejumlah video (`no_sequences_to_add`) untuk setiap gestur (`actions`). Setiap video terdiri dari 30 frame (`sequence_length`) yang disimpan sebagai file `.jpg` individual ke dalam folder `MP_Data`.

* **`2_proccess_data.py`**
    * **Tujuan:** Mengonversi gambar `.jpg` mentah menjadi data *keypoints* (titik-titik sendi tangan) yang siap dilatih.
    * **Cara Kerja:** Menggunakan `mp.solutions.holistic` dari MediaPipe untuk mengekstrak 63 koordinat (21 landmark x 3 koordinat x/y/z) dari tangan kanan (`right_hand_landmarks`) dari setiap frame gambar. Hasilnya (sebuah array dengan *shape* `(30, 63)`) disimpan sebagai file `.npy` di dalam folder `Keypoints_Data`.

* **`3_train_model.py`**
    * **Tujuan:** Melatih model AI (LSTM) untuk mengenali pola dari data *keypoints*.
    * **Cara Kerja:** Memuat semua file `.npy` dari `Keypoints_Data`. Data dibagi menjadi data latih dan data uji (80/20). Sebuah model `Sequential` Keras dibangun menggunakan layer **LSTM** (yang ideal untuk data sekuensial/temporal) dan layer `Dropout` (untuk mencegah *overfitting*). Skrip ini menggunakan `ModelCheckpoint` untuk menyimpan **hanya versi terbaik** dari model (`hand_gesture_model_terbaik.keras`) berdasarkan performanya.

* **`4_convert_to_tflite.py`**
    * **Tujuan:** Mengubah model `.keras` (yang berat) menjadi file `.tflite` (yang ringan dan *portable*).
    * **Cara Kerja:** Memuat `hand_gesture_model_terbaik.keras` dan menggunakan `TFLiteConverter.from_concrete_functions` dengan *input shape* statis `(1, 30, 63)`. Ini "membekukan" model, membuatnya sangat efisien dan kompatibel untuk inferensi.
    * **Hasil:** File `model.tflite`.

* **`5_test_model_live.py`**
    * **Tujuan:** Skrip uji coba di Windows untuk memvalidasi model `tflite` dan komunikasi serial ke ESP32 (menggunakan `pyserial` di port `COM5`).

### 2. `gesture_docker/` (Alur Deployment)

* **`Dockerfile`**
    * **Tujuan:** Resep untuk membangun "paket" aplikasi independen.
    * **Cara Kerja:**
        1.  Mulai dari `python:3.11-slim-bookworm` (OS Linux ARM64).
        2.  Menginstal semua *library* sistem (`libgl1`, `libglib2.0-0`, `libxcb1`, dll.) yang dibutuhkan oleh OpenCV (`cv2.imshow`) agar bisa memunculkan jendela GUI.
        3.  Menginstal 3 paket Python dari `requirements.txt`: `tensorflow`, `opencv-python`, dan `mediapipe`
        4.  Menyalin skrip `run_model.py` dan `model.tflite` ke dalam image.
        5.  Mengatur perintah default untuk menjalankan `run_model.py` saat container dimulai.

* **`requirements.txt`**
    * **Tujuan:** Memberi tahu Docker paket Python apa saja yang harus diinstal (`tensorflow`, `opencv-python`, `mediapipe`).

* **`run_model.py`**
    * **Tujuan:** Ini adalah skrip utama yang berjalan **di dalam Docker**.
    * **Cara Kerja:** Ini adalah versi modifikasi dari `5_test_model_live.py` di mana **semua kode serial telah dihapus**. Skrip ini memuat `model.tflite` , mengambil gambar dari webcam, melakukan deteksi, dan menampilkan hasilnya di jendela OpenCV.

---

## 🚀 Cara Penggunaan (Menjalankan di Device Target)

Panduan ini mengasumsikan Anda akan menggunakan image Docker yang sudah jadi (`azariafwn/gesture-app:latest`) yang di-build dari repo ini.

**Prasyarat di Device Target (misal: Raspberry Pi 5):**
* Device terhubung ke internet.
* Webcam terpasang.
* Docker sudah terinstal.

### 1. Tahapan Awal (Setup Pertama Kali)

1.  **Tarik Image Docker:**
    Buka terminal di Raspberry Pi Anda dan tarik image dari Docker Hub.
    ```bash
    docker pull azariafwn/gesture-app:latest
    ```

2.  **Beri Izin Akses Layar (X11):**
    Ini **WAJIB** agar container Docker bisa "menggambar" jendela `cv2.imshow` di layar Anda. Perintah ini harus dijalankan setiap kali device di-reboot.
    ```bash
    xhost +
    ```

3.  **(Rekomendasi) Buat file `docker-compose.yml`:**
    Agar Anda tidak perlu mengetik perintah `docker run` yang panjang setiap saat, buat satu folder (misal `~/gesture_app`) dan di dalamnya buat file `docker-compose.yml` dengan isi berikut:

    ```yaml
    version: '3.8'
    services:
      gesture-app:
        image: azariafwn/gesture-app:latest
        stdin_open: true # Sama dengan -i
        tty: true        # Sama dengan -t
        privileged: true # Memberi akses ke device (seperti /dev/video0)
        environment:
          - DISPLAY=${DISPLAY}
        volumes:
          - /tmp/.X11-unix:/tmp/.X11-unix
    ```

4.  **Jalankan Aplikasi:**
    Masuk ke folder tempat Anda menyimpan `docker-compose.yml` dan jalankan:
    ```bash
    docker compose up
    ```
    Jendela OpenCV akan muncul dan model akan mulai mendeteksi. Tekan `Ctrl+C` di terminal untuk menghentikannya.

### 2. Tahapan Ulang (Menjalankan Kembali)

Jika Anda sudah pernah melakukan setup, Anda hanya perlu:

1.  Buka terminal.
2.  Beri izin GUI (jika baru reboot): `xhost +`
3.  Masuk ke folder `docker-compose.yml` Anda.
4.  Jalankan: `docker compose up`

### 3. Tahapan Update (Jika Ada Model/Kode Baru)

Jika saya (pemilik repo) meng-update kode dan mem-build image baru ke `azariafwn/gesture-app:latest`, Anda hanya perlu melakukan ini di Raspberry Pi:

1.  Tarik versi terbaru:
    ```bash
    docker pull azariafwn/gesture-app:latest
    ```
2.  Jalankan seperti biasa (langkah 2 di atas).

---

## 🛠️ Alur Kerja Development (Melatih & Build Model Sendiri)

Gunakan alur ini jika Anda ingin melatih model Anda sendiri (misal: menambah gestur baru).

### Bagian A: Melatih Model (di PC Windows)

1.  Pastikan Anda memiliki *environment* Python (disarankan `venv`) dengan `tensorflow`, `opencv-python`, `mediapipe`, dan `scikit-learn`.
2.  Masuk ke folder `LSTM_HandGesture`.
3.  [cite_start]Sesuaikan `actions = np.array([...])` di `1_collect_data.py` [cite: 1] [cite_start]dan `2_proccess_data.py` [cite: 2] agar sesuai dengan gestur baru Anda.
4.  Jalankan `1_collect_data.py` untuk merekam data baru Anda.
5.  Jalankan `2_proccess_data.py` untuk memproses data.
6.  [cite_start]Jalankan `3_train_model.py` untuk melatih model[cite: 3].
7.  [cite_start]Jalankan `4_convert_to_tflite.py` untuk membuat `model.tflite`[cite: 4].

### Bagian B: Mem-build Image Docker Baru

1.  Salin `model.tflite` baru dari `LSTM_HandGesture` ke `gesture_docker`.
2.  **PENTING:** Edit baris `actions = ...` di `gesture_docker/run_model.py` agar cocok dengan gestur baru Anda.
3.  Masuk ke folder `gesture_docker`.
4.  Jalankan `docker buildx` untuk mem-build dan mem-push ke Docker Hub Anda:
    ```bash
    # Ganti 'usernameanda/nama-app' dengan nama image Anda
    docker buildx build --platform linux/arm64 -t usernameanda/gesture-app:latest --push .
    ```
5.  Setelah selesai, Anda bisa men-`pull` image baru ini di Raspberry Pi Anda.
