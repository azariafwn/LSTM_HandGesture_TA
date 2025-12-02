import numpy as np
import tensorflow as tf
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Load Model & Data
model = tf.keras.models.load_model('hand_gesture_model_terbaik.keras')
DATA_PATH = os.path.join('Keypoints_Data') 
actions = np.array(['close_to_open_palm', 'open_to_close_palm', 'close_to_one', 'open_to_one', 'close_to_two', 'open_to_two', 'close_to_three', 'open_to_three', 'close_to_four', 'open_to_four'])
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    for file in os.listdir(action_path):
        res = np.load(os.path.join(action_path, file))
        sequences.append(res)
        labels.append(label_map[action])

X = np.array(sequences)
y_true = np.array(labels)

# Prediksi
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)

# Buat Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=actions, yticklabels=actions, cmap='Blues')
plt.ylabel('Label Asli')
plt.xlabel('Prediksi Model')
plt.title('Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix_result.png')
print("Gambar Confusion Matrix tersimpan!")