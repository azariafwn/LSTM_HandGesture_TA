import tensorflow as tf

# --- Baca akurasi dari file ---
try:
    with open("latest_accuracy.txt", "r") as f:
        accuracy_str = f.read()
except FileNotFoundError:
    accuracy_str = "N/A"
# -----------------------------------------------

# Tentukan spesifikasi input STATIS (Batch size=1)
INPUT_SHAPE = (1, 30, 63) 
# -------------------------------------

# Muat model .keras terbaik
model = tf.keras.models.load_model('hand_gesture_model_terbaik.keras')
print("Model .keras berhasil dimuat.")

# --- METODE KONVERSI BARU YANG LEBIH ROBUST ---
run_model = tf.function(model.call)
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(INPUT_SHAPE, model.inputs[0].dtype)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_func],
    model
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]
# -----------------------------------------------

print("Memulai konversi TFLite murni (metode concrete function + batch size statis)...")

# Lakukan konversi
tflite_model = converter.convert()

# Simpan model .tflite
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("\n-------------------------------------------------")
print("BERHASIL!")
print(f"Model TFLite MURNI (Akurasi {accuracy_str}%) berhasil dibuat.")
print("File 'model.tflite' ini sekarang 100% siap untuk Raspberry Pi.")
print("-------------------------------------------------")