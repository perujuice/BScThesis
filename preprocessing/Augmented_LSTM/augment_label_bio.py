import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
DATA_DIR = "assets/extracted_keypoints"
OUTPUT_DIR = "preprocessing/Augmented_LSTM"
TRAIN_OUT = os.path.join(OUTPUT_DIR, "squat_sequences_train.npz")
TEST_OUT = os.path.join(OUTPUT_DIR, "squat_sequences_test.npz")

# Labels
categories = {
    "dataset-good": 1,
    "good-new": 1,
    "dataset-bad": 0,
    "bad-new": 0
}

# Frame-level features to use
FEATURES = ["valgus_angle_left", "valgus_angle_right", "torso_angle", "squat_depth"]

# Augmentation functions
def add_gaussian_noise(sequence, std=0.05):
    return sequence + np.random.normal(0, std, size=sequence.shape)

def mirror_sequence(sequence):
    mirrored = sequence.copy()
    mirrored[:, 0], mirrored[:, 1] = sequence[:, 1], sequence[:, 0]  # Swap valgus L/R
    return mirrored

# Step 1: Load and label all sequences
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

# Step 2: Train/test split (only original data)
train_seq, test_seq, train_labels, test_labels = train_test_split(
    all_sequences, all_labels, test_size=0.3, stratify=all_labels, random_state=42
)

# Step 3: Fit scaler only on original training data (stacked)
scaler = MinMaxScaler()
flat_train = np.vstack(train_seq)
scaler.fit(flat_train)

# Step 4: Normalize train/test sequences BEFORE padding
train_seq_norm = [scaler.transform(seq) for seq in train_seq]
test_seq_norm = [scaler.transform(seq) for seq in test_seq]

# Step 5: Augment and normalize training sequences
aug_sequences, aug_labels = [], []

for seq, label in zip(train_seq_norm, train_labels):
    aug_sequences.append(seq)
    aug_labels.append(label)
    
    for _ in range(2):
        noisy_seq = add_gaussian_noise(seq)
        aug_sequences.append(noisy_seq)
        aug_labels.append(label)

    mirrored_seq = mirror_sequence(seq)
    aug_sequences.append(mirrored_seq)
    aug_labels.append(label)

# Step 6: Pad everything (after normalization)
max_len = max(max(len(s) for s in aug_sequences), max(len(s) for s in test_seq_norm))
X_train = pad_sequences(aug_sequences, maxlen=max_len, padding="post", dtype="float32")
X_test = pad_sequences(test_seq_norm, maxlen=max_len, padding="post", dtype="float32")
y_train = np.array(aug_labels, dtype=np.int32)
y_test = np.array(test_labels, dtype=np.int32)

# Step 7: Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez(TRAIN_OUT, X=X_train, y=y_train)
np.savez(TEST_OUT, X=X_test, y=y_test)

# Confirm
print("✅ Cleaned, normalized & padded data saved:")
print(f"   → Train samples: {X_train.shape[0]}")
print(f"   → Test samples: {X_test.shape[0]}")
print(f"   → Sequence length: {max_len}")
print(f"   → Features per frame: {X_train.shape[2]}")
