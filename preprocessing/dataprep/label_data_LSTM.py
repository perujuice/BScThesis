import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Path to your dataset
DATA_DIR = "assets/extracted_keypoints"

# Label mapping
categories = {"dataset-good": 1, "dataset-bad": 0}

X_data, y_labels, sequence_lengths = [], [], []

for category, label in categories.items():
    folder = os.path.join(DATA_DIR, category)
    
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, filename))

            # Required columns for sequence modeling
            expected_columns = {
                "valgus_angle_left", "valgus_angle_right",
                "torso_angle", "squat_depth"
            }
            if expected_columns.issubset(df.columns):
                # Use 4 features per frame: L/R valgus, torso angle, depth
                sequence = df[[
                    "valgus_angle_left",
                    "valgus_angle_right",
                    "torso_angle",
                    "squat_depth"
                ]].values.astype("float32")

                X_data.append(sequence)
                y_labels.append(label)
                sequence_lengths.append(len(sequence))

# Pad all sequences to the length of the longest one
max_len = max(sequence_lengths)
X_padded = pad_sequences(X_data, maxlen=max_len, padding="post", dtype="float32")

# Convert labels to NumPy array
y_labels = np.array(y_labels, dtype=np.int32)

# Save to .npz
np.savez("preprocessing/squat_sequences.npz", X=X_padded, y=y_labels)

print(f" Saved padded sequence data: X shape = {X_padded.shape}, y shape = {y_labels.shape}")
