import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("assets/extracted_keypoints/dataset-good/good1.csv")

plt.plot(df["frame"], df["squat_depth"], label="Squat Depth")
plt.xlabel("Frame")
plt.ylabel("Depth (Hip - Knee Y)")
plt.title("Squat Depth Over Time")
plt.legend()
plt.show()
