import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Config ---
DATA_DIR = "assets/extracted_keypoints"
OUTPUT_DIR = "preprocessing/Augmented_LSTM"
TRAIN_OUT = os.path.join(OUTPUT_DIR, "squat_sequences_train.npz")
TEST_OUT = os.path.join(OUTPUT_DIR, "squat_sequences_test.npz")

categories = {
    "dataset-good": 1,
    "good-new": 1,
    "dataset-bad": 0,
    "bad-new": 0
}

#  Correct handcrafted features to use (from your non-augmented script)
FEATURES = [
    "vknee_flexion_left",
    "knee_flexion_right",
    "valgus_ratio",
    "torso_angle",
    "squat_depth",
    "foot_width"
]

# --- Augmentation helpers ---
def add_gaussian_noise(sequence, std=0.05):
    noise = np.random.normal(0, std, size=sequence.shape)
    return sequence + noise

def mirror_sequence(sequence):
    mirrored = sequence.copy()
    # If needed, swap left/right leg features
    # Assuming index 0: left knee, index 1: right knee
    mirrored[:, 0], mirrored[:, 1] = sequence[:, 1], sequence[:, 0]
    return mirrored

# --- Load & label original sequences ---
all_sequences, all_labels = [], []

for category, label in categories.items():
    folder = os.path.join(DATA_DIR, category)
    for fname in os.listdir(folder):
        if fname.endswith(".csv"):
            path = os.path.join(folder, fname)
            df = pd.read_csv(path)

            if set(FEATURES).issubset(df.columns):
                seq = df[FEATURES].values.astype(np.float32)
                if seq.shape[0] > 0:  # Make sure sequence is not empty
                    all_sequences.append(seq)
                    all_labels.append(label)

print(f"📥 Loaded {len(all_sequences)} original sequences")

# --- Train/test split on original data only ---
train_seq, test_seq, train_labels, test_labels = train_test_split(
    all_sequences, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)

# --- Normalize (fit scaler only on train) ---
scaler = MinMaxScaler()
flat_train = np.vstack(train_seq)
scaler.fit(flat_train)

# Normalize originals before augmentation
train_seq_norm = [scaler.transform(seq) for seq in train_seq]
test_seq_norm = [scaler.transform(seq) for seq in test_seq]

# --- Augment training set ---
aug_sequences, aug_labels = [], []

for seq, label in zip(train_seq_norm, train_labels):
    aug_sequences.append(seq)
    aug_labels.append(label)

    # Add two noise-augmented versions
    for _ in range(2):
        noisy = add_gaussian_noise(seq)
        noisy = scaler.transform(noisy)  # Normalize noisy version
        aug_sequences.append(noisy)
        aug_labels.append(label)

    # Add mirrored version
    mirrored = mirror_sequence(seq)
    mirrored = scaler.transform(mirrored)
    aug_sequences.append(mirrored)
    aug_labels.append(label)

print(f"🧪 Training set: {len(train_seq)} original → {len(aug_sequences)} after augmentation")

# --- Padding ---
max_len = max(max(len(seq) for seq in aug_sequences), max(len(seq) for seq in test_seq_norm))

X_train = pad_sequences(aug_sequences, maxlen=max_len, padding="post", dtype=np.float32)
X_test = pad_sequences(test_seq_norm, maxlen=max_len, padding="post", dtype=np.float32)
y_train = np.array(aug_labels, dtype=np.int32)
y_test = np.array(test_labels, dtype=np.int32)

# --- Save ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez(TRAIN_OUT, X=X_train, y=y_train)
np.savez(TEST_OUT, X=X_test, y=y_test)

# --- Done ---
print("✅ Augmented, normalized & padded data saved:")
print(f"   → Train samples: {X_train.shape}")
print(f"   → Test samples: {X_test.shape}")
print(f"   → Sequence length: {max_len}")
print(f"   → Features per frame: {X_train.shape[2]}")
