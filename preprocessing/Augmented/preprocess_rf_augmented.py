import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Paths
DATA_DIR = "assets/extracted_keypoints_raw"
OUTPUT_DIR = "preprocessing"
TRAIN_CSV = os.path.join(OUTPUT_DIR, "squat_dataset_train_augmented_and_original_normalized.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "squat_dataset_test_original_normalized.csv")

categories = {"dataset-good": 1, "dataset-bad": 0}
JOINTS = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_shoulder", "right_shoulder"
]
expected_angles = {
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "left_ankle_angle", "right_ankle_angle",
    "trunk_angle"
}

# Augmentation functions
def add_gaussian_noise(df, std=0.01):
    noisy = df.copy()
    coord_cols = [col for col in df.columns if any(j in col for j in JOINTS) and col.endswith(("_x", "_y", "_z"))]
    noisy[coord_cols] += np.random.normal(0, std, size=noisy[coord_cols].shape)
    angle_cols = list(expected_angles)
    noisy[angle_cols] += np.random.normal(0, 1.5, size=noisy[angle_cols].shape)
    return noisy

def mirror_sequence(df):
    mirrored = df.copy()
    for joint_l, joint_r in zip(JOINTS[::2], JOINTS[1::2]):
        for axis in ["x", "y", "z"]:
            col_l, col_r = f"{joint_l}_{axis}", f"{joint_r}_{axis}"
            mirrored[[col_l, col_r]] = mirrored[[col_r, col_l]].values
    for angle_l, angle_r in zip(
        ["left_knee_angle", "left_hip_angle", "left_ankle_angle"],
        ["right_knee_angle", "right_hip_angle", "right_ankle_angle"]
    ):
        mirrored[[angle_l, angle_r]] = mirrored[[angle_r, angle_l]].values
    return mirrored

def extract_summary_features(df):
    features = {}
    for joint in JOINTS:
        for axis in ["x", "y", "z"]:
            col = f"{joint}_{axis}"
            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_max"] = df[col].max()
            features[f"{col}_min"] = df[col].min()
    for angle in expected_angles:
        features[f"{angle}_mean"] = df[angle].mean()
        features[f"{angle}_max"] = df[angle].max()
        features[f"{angle}_min"] = df[angle].min()
    return features

# Step 1: Gather all file paths and labels
all_files = []
for category, label in categories.items():
    folder = os.path.join(DATA_DIR, category)
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            all_files.append((os.path.join(folder, filename), label))

# Step 2: Split file paths into train/test (stratified)
train_files, test_files = train_test_split(
    all_files, test_size=0.3, stratify=[lbl for _, lbl in all_files], random_state=42
)

# Step 3: Process training files (original + augmented)
train_rows = []
for path, label in train_files:
    df = pd.read_csv(path)
    if not all(f"{joint}_{axis}" in df.columns for joint in JOINTS for axis in ["x", "y", "z"]): continue
    if not expected_angles.issubset(df.columns): continue

    # Original
    row = extract_summary_features(df)
    row["label"] = label
    train_rows.append(row)

    # Augmented
    for _ in range(3):
        row = extract_summary_features(add_gaussian_noise(df))
        row["label"] = label
        train_rows.append(row)

    row = extract_summary_features(mirror_sequence(df))
    row["label"] = label
    train_rows.append(row)

# Step 4: Process test files (original only)
test_rows = []
for path, label in test_files:
    df = pd.read_csv(path)
    if not all(f"{joint}_{axis}" in df.columns for joint in JOINTS for axis in ["x", "y", "z"]): continue
    if not expected_angles.issubset(df.columns): continue
    row = extract_summary_features(df)
    row["label"] = label
    test_rows.append(row)

# Step 5: Normalize using only training data stats
train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)

X_train = train_df.drop(columns="label")
y_train = train_df["label"]
X_test = test_df.drop(columns="label")
y_test = test_df["label"]

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_final["label"] = y_train.values

test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_final["label"] = y_test.values

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
train_final.to_csv(TRAIN_CSV, index=False)
test_final.to_csv(TEST_CSV, index=False)

print(f"✅ Saved: {len(train_final)} training samples → {TRAIN_CSV}")
print(f"✅ Saved: {len(test_final)} test samples → {TEST_CSV}")