import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Path to your dataset
DATA_DIR = "assets/extracted_keypoints_raw"

# Label mapping: 1 for good squat, 0 for bad squat
categories = {"dataset-good": 1, "dataset-bad": 0}

X_data, y_labels, sequence_lengths = [], [], []

# These are the angle columns we'll use
angle_columns = [
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "left_ankle_angle", "right_ankle_angle",
    "trunk_angle"
]

# Step 1: Load all sequences (unnormalized)
for category, label in categories.items():
    folder = os.path.join(DATA_DIR, category)

    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, filename))

            if set(angle_columns).issubset(df.columns):
                sequence = df[angle_columns].values.astype("float32")
                X_data.append(sequence)
                y_labels.append(label)
                sequence_lengths.append(len(sequence))

# Step 2: Stack all data to compute min/max per feature
all_data_stacked = np.vstack(X_data)  # shape: (total_frames, 7)
feature_min = all_data_stacked.min(axis=0)  # shape: (7,)
feature_max = all_data_stacked.max(axis=0)

# Avoid division by zero
feature_range = np.where(feature_max - feature_min == 0, 1.0, feature_max - feature_min)

# Step 3: Normalize each sequence using min-max scaling
X_data_normalized = [
    (seq - feature_min) / feature_range
    for seq in X_data
]

# Step 4: Pad sequences to the max length (post-padding)
max_len = max(sequence_lengths)
X_padded = pad_sequences(X_data_normalized, maxlen=max_len, padding="post", dtype="float32")

# Convert labels to NumPy array
y_labels = np.array(y_labels, dtype=np.int32)

# Save normalized data
np.savez("preprocessing/squat_sequences_angles_normalized.npz", X=X_padded, y=y_labels)

# Also save min and max for future use (e.g., during inference)
np.savez("preprocessing/angle_scaling_params.npz", min=feature_min, max=feature_max)

print(f"✅ Saved normalized sequence data:")
print(f"   X shape = {X_padded.shape} (samples, sequence_length, 7)")
print(f"   y shape = {y_labels.shape} (samples,)")
print(f"📊 Min values: {feature_min}")
print(f"📈 Max values: {feature_max}")
