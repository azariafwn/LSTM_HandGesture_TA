# GANTI DARI buster KE bullseye (Debian 11) agar apt-get update jalan
FROM python:3.9-slim-bullseye

# Install dependency sistem yang dibutuhkan OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements dan install
COPY requirements.txt .
# Tambahkan --upgrade pip agar aman
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode
COPY . .

# Expose port Flask
EXPOSE 5000

# Perintah menjalankan aplikasi
CMD ["python", "app.py"]