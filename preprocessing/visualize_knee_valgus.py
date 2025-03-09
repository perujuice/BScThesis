import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("assets/extracted_keypoints/dataset-good/good1.csv")


plt.plot(df["frame"], df["knee_valgus_ratio"], label="Knee Valgus Ratio")
plt.xlabel("Frame")
plt.ylabel("Ratio (Lower is worse)")
plt.title("Knee Valgus Ratio Over Time")
plt.legend()
plt.show()
