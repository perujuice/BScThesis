import os
import pandas as pd

# Paths to processed keypoint data
DATA_DIR = "assets/extracted_keypoints"

# Define dataset categories
categories = {"dataset-good": 1, "dataset-bad": 0}  # 1 = good squat, 0 = bad squat

# Initialize an empty list to store all rows
all_data = []

for category, label in categories.items():
    category_path = os.path.join(DATA_DIR, category)

    for filename in os.listdir(category_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(category_path, filename)

            # Read CSV file
            df = pd.read_csv(file_path)

            # Ensure required columns exist
            if {"knee_valgus_ratio", "torso_angle", "squat_depth"}.issubset(df.columns):
                # Feature engineering: Extract more meaningful squat phase statistics
                feature_set = {
                    "knee_valgus_mean": df["knee_valgus_ratio"].mean(),
                    "knee_valgus_max": df["knee_valgus_ratio"].max(),
                    "knee_valgus_min": df["knee_valgus_ratio"].min(),
                    
                    "torso_angle_mean": df["torso_angle"].mean(),
                    "torso_angle_max": df["torso_angle"].max(),
                    "torso_angle_min": df["torso_angle"].min(),

                    "squat_depth_mean": df["squat_depth"].mean(),
                    "squat_depth_max": df["squat_depth"].max(),
                    "squat_depth_min": df["squat_depth"].min(),

                    "label": label  # Assign squat label (Good = 1, Bad = 0)
                }

                # Append to dataset
                all_data.append(feature_set)

# Convert to DataFrame
final_df = pd.DataFrame(all_data)

# Save to CSV for training models
final_df.to_csv("squat_dataset.csv", index=False)

print(f"✅ Successfully processed {len(final_df)} squat samples into squat_dataset.csv!")
