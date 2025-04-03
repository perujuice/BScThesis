import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Paths
DATA_DIR = "assets/extracted_keypoints"
OUTPUT_DIR = "preprocessing\Augmented_rf\RF_Bio"
TRAIN_CSV = os.path.join(OUTPUT_DIR, "squat_handcrafted_train_augmented_normalized.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "squat_handcrafted_test_original_normalized.csv")

categories = {
    "dataset-good": 1,
    "good-new": 1,
    "dataset-bad": 0,
    "bad-new": 0
}

# --- Augmentation Functions ---
def augment_with_noise(row, std=1.0):
    noisy = row.copy()
    for col in row.index:
        if col != "label":
            noisy[col] = noisy[col] + np.random.normal(0, std)
    return noisy

def mirror_valgus_angles(row):
    mirrored = row.copy()
    for stat in ["mean", "max", "min"]:
        l_key = f"valgus_left_{stat}"
        r_key = f"valgus_right_{stat}"
        mirrored[l_key], mirrored[r_key] = row[r_key], row[l_key]
    mirrored["valgus_asymmetry"] = abs(mirrored["valgus_left_mean"] - mirrored["valgus_right_mean"])
    return mirrored

# --- Step 1: Extract all rows ---
all_rows = []
for category, label in categories.items():
    folder = os.path.join(DATA_DIR, category)
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            path = os.path.join(folder, filename)
            df = pd.read_csv(path)

            if {"valgus_angle_left", "valgus_angle_right", "torso_angle", "squat_depth"}.issubset(df.columns):
                row = {
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
                all_rows.append(row)

# --- Step 2: Split into train/test ---
df_all = pd.DataFrame(all_rows)
train_df, test_df = train_test_split(df_all, test_size=0.3, stratify=df_all["label"], random_state=42)

# --- Step 3: Augment training data ---
augmented_rows = []
for _, row in train_df.iterrows():
    original = row.to_dict()
    augmented_rows.append(original)

    # Add 3 noisy variants
    for _ in range(3):
        augmented_rows.append(augment_with_noise(row))

    # Add mirrored variant
    mirrored = mirror_valgus_angles(row)
    augmented_rows.append(mirrored)

train_aug_df = pd.DataFrame(augmented_rows)

# --- Step 4: Normalize
X_train = train_aug_df.drop(columns="label")
y_train = train_aug_df["label"]
X_test = test_df.drop(columns="label")
y_test = test_df["label"]

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_final["label"] = y_train.values

test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_final["label"] = y_test.values

# --- Step 5: Save ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_final.to_csv(TRAIN_CSV, index=False)
test_final.to_csv(TEST_CSV, index=False)

print(f"✅ Saved: {len(train_final)} training samples → {TRAIN_CSV}")
print(f"✅ Saved: {len(test_final)} test samples → {TEST_CSV}")
