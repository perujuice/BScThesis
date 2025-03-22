import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths to processed keypoint data
DATA_DIR = "assets/extracted_keypoints"

# Define dataset categories
categories = {"dataset-good": 1, "dataset-bad": 0}  # 1 = good squat, 0 = bad squat

X_data, y_labels = [], []

for category, label in categories.items():
    category_path = os.path.join(DATA_DIR, category)

    for filename in os.listdir(category_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(category_path, filename)

            # Read CSV file
            df = pd.read_csv(file_path)

            # Ensure expected columns exist
            if {"knee_valgus_ratio", "torso_angle", "squat_depth"}.issubset(df.columns):
                # Convert DataFrame to NumPy array (sequence of frames)
                squat_sequence = df[["knee_valgus_ratio", "torso_angle", "squat_depth"]].values

                # Store the sequence and corresponding label
                X_data.append(squat_sequence)
                y_labels.append(label)

# Convert lists to NumPy arrays
X_data = np.array(X_data, dtype=np.float32)
y_labels = np.array(y_labels, dtype=np.int32)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

print(f"✅ Data ready! X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
