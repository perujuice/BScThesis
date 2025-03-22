import cv2
import mediapipe as mp
import pandas as pd
import os

# Paths
RAW_DATA_DIR = "assets/raw_data"
EXTRACTED_KEYPOINTS_DIR = "assets/extracted_keypoints"
OUTPUT_VIDEO_DIR = "assets/output_videos"  # Directory to save the output video

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def visualize_and_save_landmarks_as_mp4(video_path, keypoints_csv, output_path, slow_factor=2, loop_count=5):
    """Process video, draw pose landmarks, overlay extracted features, slow it down, and save as MP4."""

    # Read extracted keypoints
    df = pd.read_csv(keypoints_csv)

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) // slow_factor  # Slow down by reducing FPS

    # Initialize VideoWriter to save the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frames = []  # Store frames for looping
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= len(df):
            break  # Stop when video ends or all frames are processed

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

        # Add the frame multiple times to slow down the video
        for _ in range(slow_factor):
            frames.append(frame)

        frame_count += 1

    # Loop the video by appending the frames multiple times
    for _ in range(loop_count):
        for frame in frames:
            out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"✅ MP4 video saved at: {output_path}")

# Function to process and save a sample squat video as MP4
def save_sample_squat_video():
    """Choose a sample squat video, process it, and save the output as MP4."""
    sample_video = os.path.join(RAW_DATA_DIR, "dataset-good", "good1.mov")  # Change filename if needed
    sample_csv = os.path.join(EXTRACTED_KEYPOINTS_DIR, "dataset-good", "good1.csv")  # Adjust filename
    output_video = os.path.join(OUTPUT_VIDEO_DIR, "good1_processed.mp4")  # Output file path

    # Ensure output directory exists
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

    if os.path.exists(sample_video) and os.path.exists(sample_csv):
        try:
            visualize_and_save_landmarks_as_mp4(sample_video, sample_csv, output_video, slow_factor=2, loop_count=5)
        except KeyboardInterrupt:
            print("\n⏹️ Interrupted by user (Ctrl + C)")
            cv2.destroyAllWindows()
    else:
        print("❌ Sample video or keypoint CSV not found!")

if __name__ == "__main__":
    save_sample_squat_video()