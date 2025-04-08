import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Initialize data collector
all_data = []

for category, label in categories.items():
    category_path = os.path.join(DATA_DIR, category)

    for filename in os.listdir(category_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(category_path, filename)
            df = pd.read_csv(file_path)

            if expected_columns.issubset(df.columns):
                # Compute simple statistical features for each variable
                feature_set = {
                    "vknee_flexion_left_mean": df["vknee_flexion_left"].mean(),
                    "vknee_flexion_left_std": df["vknee_flexion_left"].std(),
                    
                    "knee_flexion_right_mean": df["knee_flexion_right"].mean(),
                    "knee_flexion_right_std": df["knee_flexion_right"].std(),
                    
                    "valgus_ratio_mean": df["valgus_ratio"].mean(),
                    "valgus_ratio_std": df["valgus_ratio"].std(),
                    
                    "torso_angle_mean": df["torso_angle"].mean(),
                    "torso_angle_std": df["torso_angle"].std(),
                    
                    "squat_depth_mean": df["squat_depth"].mean(),
                    "squat_depth_std": df["squat_depth"].std(),
                    
                    "foot_width_mean": df["foot_width"].mean(),
                    "foot_width_std": df["foot_width"].std(),

                    "label": label
                }

                all_data.append(feature_set)

# Convert to DataFrame
final_df = pd.DataFrame(all_data)

# Save processed dataset
os.makedirs("preprocessing", exist_ok=True)
final_df.to_csv("preprocessing/squat_dataset_handcrafted.csv", index=False)

print(f"✅ Handcrafted features extracted for {len(final_df)} squat samples.")
