import os
import pandas as pd

# Paths to processed keypoint data (handcrafted features)
DATA_DIR = "assets/extracted_keypoints"

# Define dataset categories
categories = {
    "dataset-good": 1,
    "good-new": 1,
    "dataset-bad": 0,
    "bad-new": 0
}

# Expected feature columns in each CSV
expected_columns = {
    "vknee_flexion_left", "knee_flexion_right",
    "valgus_ratio", "torso_angle", "squat_depth", "foot_width"
}

# Feature strategy: include min/max for features where it makes sense
use_min_max_for = {"vknee_flexion_left", "knee_flexion_right", "valgus_ratio", "torso_angle", "squat_depth"}

# Initialize data collector
all_data = []

for category, label in categories.items():
    category_path = os.path.join(DATA_DIR, category)

    for filename in os.listdir(category_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(category_path, filename)
            df = pd.read_csv(file_path)

            if expected_columns.issubset(df.columns):
                feature_set = {}

                for col in expected_columns:
                    feature_set[f"{col}_mean"] = df[col].mean()
                    feature_set[f"{col}_std"] = df[col].std()

                    if col in use_min_max_for:
                        feature_set[f"{col}_min"] = df[col].min()
                        feature_set[f"{col}_max"] = df[col].max()

                feature_set["label"] = label
                all_data.append(feature_set)

# Convert to DataFrame
final_df = pd.DataFrame(all_data)

# Save processed dataset
os.makedirs("preprocessing", exist_ok=True)
csv_path = "preprocessing/squat_dataset_handcrafted.csv"
final_df.to_csv(csv_path, index=False)

csv_path, final_df.head()