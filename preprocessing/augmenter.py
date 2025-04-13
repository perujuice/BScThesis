import numpy as np
'''
For data augmentation, we add Gaussian noise to the input sequences.

I created a class for this so that we can easily add more augmentations and
its also easy to use without having to process that data over and over again.
We can also add other augmentations like mirroring, cropping, etc.
'''


class Augmenter:
    def __init__(self, noise_std=0.05, noise_n=1, mirror=False):
        self.noise_std = noise_std
        self.noise_n = noise_n
        self.mirror = mirror

    # This function adds Gaussian noise to the input sequence.
    # And it uses the seed to make sure that the noise is controlled and reproducible.
    def add_gaussian_noise(self, seq, seed=None):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, self.noise_std, size=seq.shape)
        # Only apply noise to non-padded frames
        mask = ~np.all(seq == 0, axis=1)
        noisy_seq = seq.copy()
        noisy_seq[mask] += noise[mask]
        return np.clip(noisy_seq, 0, 1)

    def augment(self, X_train, y_train, base_seed=0):
        X_aug, y_aug = [], []
        for i, (seq, label) in enumerate(zip(X_train, y_train)):
            X_aug.append(seq)
            y_aug.append(label)

            # Add noise_n Gaussian noise augmentations
            for j in range(self.noise_n):
                seed = base_seed * 100 + i * 10 + j # Unique seed for each augmentation
                noisy = self.add_gaussian_noise(seq, seed=seed)
                X_aug.append(noisy)
                y_aug.append(label)

        return np.array(X_aug, dtype=np.float32), np.array(y_aug, dtype=np.int32)
    
# Helper function to augment RF data
def augment_rf_data(X_rf, y_rf, augmenter, base_seed=0):
    X_rf_seq = [x.reshape(1, -1) for x in X_rf]
    X_aug_seq, y_aug = augmenter.augment(X_rf_seq, y_rf, base_seed=base_seed)
    X_aug_flat = X_aug_seq.squeeze(axis=1)
    return X_aug_flat, y_aug


