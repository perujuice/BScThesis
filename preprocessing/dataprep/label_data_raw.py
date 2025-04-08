import os
import pandas as pd

# Paths to processed keypoint data
DATA_DIR = "assets/extracted_keypoints_raw"

# Define dataset categories and labels
categories = {
    "dataset-good": 1,
    "good-new": 1,
    "dataset-bad": 0,
    "bad-new": 0
}

# Joints to extract
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
    "left_ankle_angle", "right_ankle_angle"
}

# Collect all feature rows
all_data = []

for category, label in categories.items():
    category_path = os.path.join(DATA_DIR, category)

    for filename in os.listdir(category_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(category_path, filename)
            df = pd.read_csv(file_path)

            # Verify expected columns
            if all(
                f"{joint}_{axis}" in df.columns
                for joint in JOINTS
                for axis in ["x", "y", "z"]
            ) and expected_angles.issubset(df.columns):

                feature_set = {}

                # Joint coordinate features
                for joint in JOINTS:
                    for axis in ["x", "y", "z"]:
                        col = f"{joint}_{axis}"
                        feature_set[f"{col}_mean"] = df[col].mean()
                        feature_set[f"{col}_std"] = df[col].std()
                        feature_set[f"{col}_min"] = df[col].min()
                        feature_set[f"{col}_max"] = df[col].max()

                # Joint angle features
                for angle in expected_angles:
                    feature_set[f"{angle}_mean"] = df[angle].mean()
                    feature_set[f"{angle}_std"] = df[angle].std()
                    feature_set[f"{angle}_min"] = df[angle].min()
                    feature_set[f"{angle}_max"] = df[angle].max()

                # Add label
                feature_set["label"] = label
                all_data.append(feature_set)

# Convert to DataFrame
final_df = pd.DataFrame(all_data)

# Save final dataset (non-normalized)
os.makedirs("preprocessing", exist_ok=True)
final_df.to_csv("preprocessing/squat_dataset_3d_raw.csv", index=False)

print(f"✅ Processed and saved {final_df.shape[0]} squat samples to squat_dataset_3d_raw.csv")
