import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

# Path to your dataset
DATA_DIR = "assets/extracted_keypoints_raw"

# Label mapping: 1 for good squat, 0 for bad squat
categories = {
    "dataset-good": 1,
    "good-new": 1,
    "dataset-bad": 0,
    "bad-new": 0
}

X_data, y_labels, sequence_lengths = [], [], []

# Step 1: Load all sequences and normalize independently
for category, label in categories.items():
    folder = os.path.join(DATA_DIR, category)

    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, filename))

            # Extract all features (excluding the label column if it exists)
            feature_columns = [col for col in df.columns if col != "label"]
            sequence = df[feature_columns].values.astype("float32")

            # Normalize each sequence independently
            scaler = MinMaxScaler()
            sequence_normalized = scaler.fit_transform(sequence)

            X_data.append(sequence_normalized)
            y_labels.append(label)
            sequence_lengths.append(len(sequence))

# Step 2: Pad sequences to the max length (post-padding)
max_len = max(sequence_lengths)
X_padded = pad_sequences(X_data, maxlen=max_len, padding="post", dtype="float32")

# Convert labels to NumPy array
y_labels = np.array(y_labels, dtype=np.int32)

# Save normalized and padded data
os.makedirs("preprocessing", exist_ok=True)
np.savez("preprocessing/squat_sequences_all_features_normalized.npz", X=X_padded, y=y_labels)

print(f" Saved normalized sequence data:")
print(f"   X shape = {X_padded.shape} (samples, sequence_length, num_features)")
print(f"   y shape = {y_labels.shape} (samples,)")
