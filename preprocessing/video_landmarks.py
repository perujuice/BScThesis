import cv2
import mediapipe as mp
import pandas as pd
import os

# Paths
RAW_DATA_DIR = "assets/raw_data"
EXTRACTED_KEYPOINTS_DIR = "assets/extracted_keypoints"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def visualize_landmarks(video_path, keypoints_csv):
    """Process video in real-time, draw pose landmarks & overlay extracted features."""

    # Read extracted keypoints
    df = pd.read_csv(keypoints_csv)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= len(df):
            break

        # Convert to RGB (required for MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw landmarks using MediaPipe (Live Detection)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract features for this frame (From CSV)
        frame_data = df.iloc[frame_count]
        knee_valgus = frame_data["knee_valgus_ratio"]
        torso_angle = frame_data["torso_angle"]
        squat_depth = frame_data["squat_depth"]

        # Overlay extracted features
        cv2.putText(frame, f"Knee Valgus: {knee_valgus:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Torso Angle: {torso_angle:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Squat Depth: {squat_depth:.2f}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display video
        cv2.imshow("Squat Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Visualization complete for {video_path}")

# Function to visualize a sample squat video
def visualize_sample_squat():
    """Choose a sample squat video and visualize its keypoints & features."""
    sample_video = os.path.join(RAW_DATA_DIR, "dataset-good", "good1.mov")  # Change filename if needed
    sample_csv = os.path.join(EXTRACTED_KEYPOINTS_DIR, "dataset-good", "good1.csv")  # Adjust filename
    
    if os.path.exists(sample_video) and os.path.exists(sample_csv):
        visualize_landmarks(sample_video, sample_csv)
    else:
        print("❌ Sample video or keypoint CSV not found!")

if __name__ == "__main__":
    visualize_sample_squat()
