import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Paths to processed keypoint data
DATA_DIR = "assets/extracted_keypoints_raw"

# Define dataset categories
categories = {"dataset-good": 1, "dataset-bad": 0}

# Joints to process (as in your CSVs)
JOINTS = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_shoulder", "right_shoulder"
]

# Expected angle columns
expected_angles = {
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "left_ankle_angle", "right_ankle_angle",
    "trunk_angle"
}

# Collect all feature rows
all_data = []

for category, label in categories.items():
    category_path = os.path.join(DATA_DIR, category)

    for filename in os.listdir(category_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(category_path, filename)
            df = pd.read_csv(file_path)

            # Verify expected columns exist
            if all(
                f"{joint}_{axis}" in df.columns
                for joint in JOINTS
                for axis in ["x", "y", "z"]
            ) and expected_angles.issubset(df.columns):

                feature_set = {}

                # Joint coordinate features (mean, max, min)
                for joint in JOINTS:
                    for axis in ["x", "y", "z"]:
                        col = f"{joint}_{axis}"
                        feature_set[f"{col}_mean"] = df[col].mean()
                        feature_set[f"{col}_max"] = df[col].max()
                        feature_set[f"{col}_min"] = df[col].min()

                # Joint angle features (mean, max, min)
                for angle in expected_angles:
                    feature_set[f"{angle}_mean"] = df[angle].mean()
                    feature_set[f"{angle}_max"] = df[angle].max()
                    feature_set[f"{angle}_min"] = df[angle].min()

                # Add label
                feature_set["label"] = label
                all_data.append(feature_set)

# Convert to DataFrame
final_df = pd.DataFrame(all_data)

# Separate features and labels
features = final_df.drop(columns="label")
labels = final_df["label"]

# Normalize features
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)

# Recombine
normalized_df = pd.DataFrame(features_normalized, columns=features.columns)
normalized_df["label"] = labels.values

# Save final normalized dataset
os.makedirs("preprocessing", exist_ok=True)
normalized_df.to_csv("preprocessing/squat_dataset_3d_raw_normalized.csv", index=False)

print(f"✅ Processed and normalized {normalized_df.shape[0]} samples to squat_dataset_3d_raw_normalized.csv")
