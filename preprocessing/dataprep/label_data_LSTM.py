import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

# Paths
DATA_DIR = "assets/extracted_keypoints"
OUTPUT_PATH = "preprocessing/squat_sequences_normalized.npz"

# Label mapping
categories = {
    "dataset-good": 1,
    "good-new": 1,
    "dataset-bad": 0,
    "bad-new": 0
}

X_data, y_labels, sequence_lengths = [], [], []

# Step 1: Load sequences
for category, label in categories.items():
    folder = os.path.join(DATA_DIR, category)
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, filename))

            expected_columns = {
                "valgus_angle_left", "valgus_angle_right",
                "torso_angle", "squat_depth"
            }

            if expected_columns.issubset(df.columns):
                sequence = df[[
                    "valgus_angle_left",
                    "valgus_angle_right",
                    "torso_angle",
                    "squat_depth"
                ]].values.astype("float32")

                X_data.append(sequence)
                y_labels.append(label)
                sequence_lengths.append(len(sequence))

# Step 2: Normalize using MinMaxScaler
print("📏 Normalizing sequences...")
all_frames = np.concatenate(X_data, axis=0)
scaler = MinMaxScaler()
scaler.fit(all_frames)
X_normalized = [scaler.transform(seq) for seq in X_data]

# Step 3: Pad sequences
max_len = max(sequence_lengths)
X_padded = pad_sequences(X_normalized, maxlen=max_len, padding="post", dtype="float32")
y_labels = np.array(y_labels, dtype=np.int32)

# Step 4: Save to .npz
np.savez(OUTPUT_PATH, X=X_padded, y=y_labels)
print(f"✅ Saved normalized sequence data to {OUTPUT_PATH}")
print(f"   X shape = {X_padded.shape} (samples, seq_len, features)")
print(f"   y shape = {y_labels.shape} (labels)")
