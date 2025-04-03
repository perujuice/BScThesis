import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
DATA_DIR = "assets/extracted_keypoints"
OUTPUT_DIR = "preprocessing\Augmented_LSTM"
TRAIN_OUT = os.path.join(OUTPUT_DIR, "squat_sequences_train.npz")
TEST_OUT = os.path.join(OUTPUT_DIR, "squat_sequences_test.npz")

# Labels
categories = {"dataset-good": 1, "dataset-bad": 0}

# Frame-level features
FEATURES = ["valgus_angle_left", "valgus_angle_right", "torso_angle", "squat_depth"]

# Augmentations
def add_gaussian_noise(sequence, std=1.0):
    return sequence + np.random.normal(0, std, size=sequence.shape)

def mirror_sequence(sequence):
    mirrored = sequence.copy()
    mirrored[:, 0], mirrored[:, 1] = sequence[:, 1], sequence[:, 0]  # Swap L/R valgus
    return mirrored

# Step 1: Load sequences
all_sequences = []
all_labels = []

for category, label in categories.items():
    folder = os.path.join(DATA_DIR, category)
    for fname in os.listdir(folder):
        if fname.endswith(".csv"):
            path = os.path.join(folder, fname)
            df = pd.read_csv(path)

            if set(FEATURES).issubset(df.columns):
                sequence = df[FEATURES].values.astype("float32")
                all_sequences.append(sequence)
                all_labels.append(label)

# Step 2: Train/test split (original data only)
train_seq, test_seq, train_labels, test_labels = train_test_split(
    all_sequences, all_labels, test_size=0.3, stratify=all_labels, random_state=42
)

# Step 3: Augment training data
aug_sequences, aug_labels = [], []

for seq, label in zip(train_seq, train_labels):
    aug_sequences.append(seq)  # Original
    aug_labels.append(label)

    for _ in range(2):
        aug_sequences.append(add_gaussian_noise(seq))
        aug_labels.append(label)

    aug_sequences.append(mirror_sequence(seq))
    aug_labels.append(label)

# Step 4: Padding
max_len = max(max(len(s) for s in aug_sequences), max(len(s) for s in test_seq))

X_train = pad_sequences(aug_sequences, maxlen=max_len, padding="post", dtype="float32")
X_test = pad_sequences(test_seq, maxlen=max_len, padding="post", dtype="float32")
y_train = np.array(aug_labels, dtype=np.int32)
y_test = np.array(test_labels, dtype=np.int32)

# Step 5: Normalize per feature using training stats only
scaler = MinMaxScaler()
n_features = X_train.shape[2]

X_train_flat = X_train.reshape(-1, n_features)
X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)

X_test_flat = X_test.reshape(-1, n_features)
X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

# Step 6: Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez(TRAIN_OUT, X=X_train_scaled, y=y_train)
np.savez(TEST_OUT, X=X_test_scaled, y=y_test)

print(f"✅ Saved: {X_train.shape[0]} training samples → {TRAIN_OUT}")
print(f"✅ Saved: {X_test.shape[0]} test samples → {TEST_OUT}")
