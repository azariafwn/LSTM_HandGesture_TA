# Gunakan base image Python yang kompatibel dengan ARM64 (Debian Bookworm, sama seperti RPi OS)
FROM --platform=linux/arm64 python:3.11-slim-bookworm

# Tetapkan direktori kerja di dalam container
WORKDIR /app

# Install dependensi sistem LENGKAP
RUN apt-get update && apt-get install -y \
    \
    # Dependensi untuk kompilasi C (dibutuhkan lgpio/pip)
    build-essential \
    python3-dev \
    \
    # --- 3. Dependensi OpenCV/Qt (GUI) ---
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libfontconfig1 \
    libdbus-1-3 \
    libxcb1 \
    libx11-xcb1 \
    libxcb-glx0 \
    libsm6 \
    libice6 \
    libxrender1 \
    libxext6 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render0 \
    libxcb-shm0 \
    libxcb-randr0 \
    libxcb-util1 \
    libxcb-icccm4 \
    libxcb-xkb1 \
    libv4l-0 \
    \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Salin file requirements dulu (untuk caching layer Docker)
COPY requirements.txt .

# Install dependensi Python
# Pip akan otomatis mengunduh versi ARM64
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin sisa file aplikasi Anda
COPY run_model.py .
COPY model.tflite .

# Perintah default yang akan dijalankan saat container startup
CMD ["python3", "run_model.py"]