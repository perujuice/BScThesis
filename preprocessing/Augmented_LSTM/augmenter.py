import numpy as np

class Augmenter:
    def __init__(self, noise_std=0.05, noise_n=1, mirror=False):
        self.noise_std = noise_std
        self.noise_n = noise_n
        self.mirror = mirror

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
                seed = base_seed * 100 + i * 10 + j
                noisy = self.add_gaussian_noise(seq, seed=seed)
                X_aug.append(noisy)
                y_aug.append(label)

        return np.array(X_aug, dtype=np.float32), np.array(y_aug, dtype=np.int32)

