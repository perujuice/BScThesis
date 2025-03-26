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

            # Check required columns
            expected_columns = {
                "valgus_angle_left", "valgus_angle_right",
                "torso_angle", "squat_depth"
            }
            if expected_columns.issubset(df.columns):
                # Feature engineering: Summary stats for left/right valgus and asymmetry
                feature_set = {
                    "valgus_left_mean": df["valgus_angle_left"].mean(),
                    "valgus_left_max": df["valgus_angle_left"].max(),
                    "valgus_left_min": df["valgus_angle_left"].min(),

                    "valgus_right_mean": df["valgus_angle_right"].mean(),
                    "valgus_right_max": df["valgus_angle_right"].max(),
                    "valgus_right_min": df["valgus_angle_right"].min(),

                    "valgus_asymmetry": abs(df["valgus_angle_left"].mean() - df["valgus_angle_right"].mean()),

                    "torso_angle_mean": df["torso_angle"].mean(),
                    "torso_angle_max": df["torso_angle"].max(),
                    "torso_angle_min": df["torso_angle"].min(),

                    "squat_depth_mean": df["squat_depth"].mean(),
                    "squat_depth_max": df["squat_depth"].max(),
                    "squat_depth_min": df["squat_depth"].min(),

                    "label": label
                }

                all_data.append(feature_set)

# Build final DataFrame and save
final_df = pd.DataFrame(all_data)
os.makedirs("preprocessing", exist_ok=True)
final_df.to_csv("preprocessing/squat_dataset.csv", index=False)

print(f"✅ Successfully processed {len(final_df)} squat samples into squat_dataset.csv!")
