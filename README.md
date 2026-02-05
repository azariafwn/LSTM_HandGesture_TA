# Smart Home Hand Gesture Control System 🖐️🏠

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Laravel](https://img.shields.io/badge/Laravel-10-red)
![React](https://img.shields.io/badge/React-18-cyan)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Lite-orange)

A contactless smart home control system powered by **Artificial Intelligence** and **IoT**. This project utilizes a Raspberry Pi 5 to recognize hand gestures in real-time and control various home appliances (lights, fans) connected via ESP8266 microcontrollers. It features a modern, cyberpunk-styled web dashboard for remote monitoring and manual control.

🔗 **Live Dashboard:** [https://smarthome-gesture.my.id](https://smarthome-gesture.my.id)

---

## 📖 Project Overview

This system is designed to eliminate physical interaction with switches. By using computer vision and deep learning, the system detects specific hand gestures to select a device (e.g., "Lamp 1") and execute an action (e.g., "Turn On").

The system is fully containerized using Docker and exposes a secure web dashboard accessible from anywhere via the internet, allowing for real-time video streaming and latency monitoring.

### Key Features
* **Real-time Gesture Recognition:** Powered by MediaPipe and a custom TensorFlow Lite model.
* **Low Latency Control:** Optimized communication between Raspberry Pi and ESP8266 via HTTP.
* **Smart State Machine:** Prevents accidental triggers using a selection-action logic (Select Device -> Perform Action).
* **Modern Dashboard:** Built with Laravel, React, and Tailwind CSS (Dark Mode theme).
* **Multi-Client Streaming:** Supports simultaneous video monitoring from multiple devices.
* **Auto Discovery:** Uses Zeroconf/mDNS to automatically find IoT devices on the network.

---

## ⚙️ System Architecture

The system consists of three main layers: **Edge Processing (AI)**, **IoT Actuation**, and **User Interface**.

### 1. AI & Processing Layer (Raspberry Pi 5)
* **Input:** A webcam captures video frames.
* **Feature Extraction:** **MediaPipe Hands** extracts 21 3D hand landmarks.
* **Classification:** A custom **TensorFlow Lite** model processes the landmark history (sequence) to predict the gesture.
* **Logic:** A Python script manages the state (Cooldowns, Device Selection, Action Triggers).
* **Streaming:** A multi-threaded **Flask** server streams the processed video feed with overlays.

### 2. IoT Layer (ESP8266)
* **Actuators:** ESP8266 modules control relays for lights fans, and many more.
* **Communication:** Receives HTTP commands from the Raspberry Pi (e.g., `GET /11` to turn Device 1 ON).

### 3. Interface Layer (Web Dashboard)
* **Backend:** **Laravel** serves the application and manages authentication.
* **Database:** **SQLite** stores activity logs (gestures detected, latency, device status).
* **Frontend:** **React (Inertia.js)** provides a reactive UI for:
    * Live Video Monitoring.
    * Real-time Activity Logs.
    * Manual Device Control (Buttons).
    * Network & Inference Latency Stats.

---

## 🛠️ Tech Stack

### Hardware
* **Edge Device:** Raspberry Pi 5 (8GB RAM recommended)
* **Camera:** USB Webcam / Pi Camera
* **Microcontroller:** NodeMCU ESP8266 (x4)
* **Network:** Local WiFi Router

### Software & Frameworks
* **AI/ML:** Python, OpenCV, MediaPipe, TensorFlow, Keras.
* **Backend (Stream):** Flask, Zeroconf, SQLite, Threading.
* **Web Framework:** Laravel 10 (PHP), Docker.
* **Frontend:** React.js, Tailwind CSS, Inertia.js.
* **Tools:** Docker Compose, Postman (API Testing).

---

## 🤏 Gesture Logic

The system uses a **Two-Step Activation** method to ensure safety and accuracy:

1.  **Step 1: Selection (Select Device)**
    * ✌️ **Two Fingers:** Selects Device 2.
    * 🤟 **Three Fingers:** Selects Device 3.
    * (And so on for other devices).

2.  **Step 2: Action (Execute Command)**
    * ✋ **Closed to Open Palm:** Turn **ON**.
    * ✊ **Open to Closed Palm:** Turn **OFF**.

*Example: To turn on Lamp 2, the user shows "Two Fingers" followed by "Opening Palm".*

---

## 🧠 Machine Learning Architecture

The core of the gesture recognition system is a Deep Neural Network built using **TensorFlow** and **Keras**. The model is designed to classify time-series data derived from hand landmarks, enabling the detection of dynamic gestures rather than just static poses.

### 1. Input Data
The model accepts sequential data extracted from **MediaPipe Hands**:
* **Sequence Length:** 30 frames (representing ~1 second of video).
* **Features:** 63 values per frame (21 landmarks $\times$ 3 coordinates [x, y, z]).
* **Input Shape:** `(30, 63)`

### 2. Network Topology
The architecture utilizes **Long Short-Term Memory (LSTM)** layers to capture temporal dependencies and motion patterns, followed by Fully Connected (Dense) layers for classification.

| Layer Type | Units / Rate | Activation | Description |
| :--- | :--- | :--- | :--- |
| **LSTM** | 64 | `tanh` | Captures initial temporal features (return sequences). |
| **Dropout** | 0.5 | - | Regularization to prevent overfitting. |
| **LSTM** | 128 | `tanh` | Deep temporal feature extraction (return sequences). |
| **Dropout** | 0.5 | - | Regularization. |
| **LSTM** | 64 | `tanh` | Compresses time-series data into a feature vector. |
| **Dense** | 64 | `relu` | Intermediate classification logic. |
| **Dense** | 32 | `relu` | Bottleneck layer for feature refinement. |
| **Dense** | *N_Classes* | `softmax` | Output layer (Probability distribution for gestures). |

### 3. Training Configuration
The model is trained using the following hyperparameters and callbacks to ensure optimal convergence:

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Batch Size:** 32
* **Max Epochs:** 200
* **Callbacks:**
    * `EarlyStopping`: Monitors `val_loss` with a patience of 20 epochs to stop training when the model stops improving.
    * `ModelCheckpoint`: Automatically saves the best model weights based on validation loss.

---

## ⚠️ Disclaimer

This project is created for educational and research purposes (Final Year Project). While the dashboard is accessible publicly for demonstration, the physical control is limited to the local hardware setup located in the development environment.

---

<p align="center">
  Built with ❤️ by <a href="https://azariafwn.vercel.app/" target="_blank"><strong>azariafwn</strong></a>
</p>