import json
import os
from scipy.stats import wilcoxon

folder = "../statistical_significance"

model_names = {
    "BiLSTM_biomech_accuracy.json": "BiLSTM Biomech Accuracy",
    "BiLSTM_biomech_aug_accuracy.json": "BiLSTM Biomech Aug Accuracy",
    "BiLSTM_raw_accuracy.json": "BiLSTM Raw Accuracy",
    "BiLSTM_raw_aug_accuracy.json": "BiLSTM Raw Aug Accuracy",
    "rf_biomech_accuracy.json": "RF Biomech Accuracy",
    "rf_biomech_aug_accuracy.json": "RF Biomech Aug Accuracy",
    "rf_raw_accuracy.json": "RF Raw Accuracy",
    "rf_raw_aug_accuracy.json": "RF Raw Aug Accuracy",
}

# Load all accuracy lists
# Load all accuracy lists with error handling
accuracies = {}
for file in os.listdir(folder):
    path = os.path.join(folder, file)
    try:
        with open(path) as f:
            data = json.load(f)
            if not data:
                raise ValueError("File is empty")
            accuracies[file] = data
    except Exception as e:
        print(f" Skipping {file} — error: {e}")



# Define best model
best_model_file = "rf_biomech_aug_accuracy.json"
best_scores = accuracies[best_model_file]

# Perform Wilcoxon test against all others
print(f"\nComparing all models to: {model_names[best_model_file]}")
print(f"{'Model':40} | p-value   | Significant?")
print("-" * 70)

for file, scores in accuracies.items():
    if file == best_model_file:
        continue
    stat, p = wilcoxon(best_scores, scores)
    significant = "Yes" if p < 0.05 else "No"
    print(f"{model_names[file]:40} | {p:.4f}   | {significant}")