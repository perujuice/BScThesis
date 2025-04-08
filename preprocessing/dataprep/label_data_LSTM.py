import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

# Paths
DATA_DIR = "assets/extracted_keypoints"
OUTPUT_PATH = "preprocessing/ready/squat_sequences__normalized.npz"

# Label mapping
categories = {
    "dataset-good": 1,
    "good-new": 1,
    "dataset-bad": 0,
    "bad-new": 0
}

# Your handcrafted features (excluding "frame")
feature_columns = [
    "vknee_flexion_left",
    "knee_flexion_right",
    "valgus_ratio",
    "torso_angle",
    "squat_depth",
    "foot_width"
]

# Containers
X_data, y_labels, sequence_lengths = [], [], []

# Load sequences
for category, label in categories.items():
    folder = os.path.join(DATA_DIR, category)
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder, filename)
            df = pd.read_csv(file_path)

            if set(feature_columns).issubset(df.columns):
                sequence = df[feature_columns].values.astype(np.float32)
                X_data.append(sequence)
                y_labels.append(label)
                sequence_lengths.append(len(sequence))

print(f"📁 Loaded {len(X_data)} sequences.")

# Normalize globally across all frames and features
print("📏 Applying MinMax normalization...")
all_frames = np.vstack(X_data)
scaler = MinMaxScaler()
scaler.fit(all_frames)

X_normalized = [scaler.transform(seq) for seq in X_data]

# Pad sequences to the length of the longest one
max_len = max(sequence_lengths)
X_padded = pad_sequences(X_normalized, maxlen=max_len, padding="post", dtype=np.float32)
y_labels = np.array(y_labels, dtype=np.int32)

# Save
np.savez(OUTPUT_PATH, X=X_padded, y=y_labels)
print(f" Saved normalized padded data to {OUTPUT_PATH}")
print(f"    X shape: {X_padded.shape} (samples, seq_len, features)")
print(f"    y shape: {y_labels.shape} (labels)")
