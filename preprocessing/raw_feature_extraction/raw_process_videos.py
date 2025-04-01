import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

'''
The preprocessing here includes the extraction of raw features from the videos.
In this case that means extracting the raw joint coordinates and joint angles from the videos.
Since the videos were filmed at an angled view, mediapipes's pose model will work better with 3D coordinates.
The 3D coordinates are then used to calculate the angles between the joints.
'''


# Paths to datasets
RAW_DATA_DIR = "assets/raw_data"
OUTPUT_DIR = "assets/extracted_keypoints_raw"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Landmarks indices
LANDMARKS = {
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_foot": 29, "right_foot": 30,
}

'''
I'll over-explain these functions a bit just so we can understand them better.
Especially so that it'll be easier to put this information into the thesis.
'''

# Calculate angle between three points
# a, b, c are the three points
# Returns the angle in degrees
# This function calculates the angle between three points (shoulder, hip, ankle) using the cosine rule.
# The cosine rule states that the cosine of an angle in a triangle can 
# be calculated using the dot product of two vectors and their magnitudes.
# The angle is then calculated using the arccosine function. The function returns the angle in degrees.
def calculate_angle(a, b, c):
    """Calculate angle between three 3D points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


# Process video and extract keypoints
# video_path is the path to the video file
# output_csv is the path to the output CSV file
# This function processes a video file frame by frame using MediaPipe Pose.
# The extracted features are saved to a CSV file.
def process_video_raw_features(video_path, output_csv):
    """Extract raw 3D joint coordinates and joint angles from each video frame."""
    cap = cv2.VideoCapture(video_path)
    data = []
    frame_count = 0

    def get_point(idx):
        """Helper to get (x, y, z) tuple for a landmark index."""
        landmark = lm[idx]
        return (landmark.x, landmark.y, landmark.z)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            frame_data = {"frame": frame_count}
            lm = results.pose_landmarks.landmark

            # Save raw joint coordinates (x, y, z)
            for name, idx in LANDMARKS.items():
                coords = get_point(idx)
                frame_data[f"{name}_x"] = coords[0]
                frame_data[f"{name}_y"] = coords[1]
                frame_data[f"{name}_z"] = coords[2]

            # Joint angles using 3D points
            hip_left = get_point(LANDMARKS["left_hip"])
            knee_left = get_point(LANDMARKS["left_knee"])
            ankle_left = get_point(LANDMARKS["left_ankle"])
            shoulder_left = get_point(LANDMARKS["left_shoulder"])
            foot_left = get_point(LANDMARKS["left_foot"])

            hip_right = get_point(LANDMARKS["right_hip"])
            knee_right = get_point(LANDMARKS["right_knee"])
            ankle_right = get_point(LANDMARKS["right_ankle"])
            shoulder_right = get_point(LANDMARKS["right_shoulder"])
            foot_right = get_point(LANDMARKS["right_foot"])

            frame_data["left_knee_angle"] = calculate_angle(hip_left, knee_left, ankle_left)
            frame_data["right_knee_angle"] = calculate_angle(hip_right, knee_right, ankle_right)
            frame_data["left_hip_angle"] = calculate_angle(shoulder_left, hip_left, knee_left)
            frame_data["right_hip_angle"] = calculate_angle(shoulder_right, hip_right, knee_right)
            frame_data["left_ankle_angle"] = calculate_angle(knee_left, ankle_left, foot_left)
            frame_data["right_ankle_angle"] = calculate_angle(knee_right, ankle_right, foot_right)

            # Trunk angle: between shoulder–hip and vertical-down vector
            vertical_down = (hip_left[0], hip_left[1] + 0.1, hip_left[2])  # down in image space
            frame_data["trunk_angle"] = calculate_angle(shoulder_left, hip_left, vertical_down)

            data.append(frame_data)

        frame_count += 1

    cap.release()
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Processed {video_path} -> {output_csv} (3D raw features)")


# Process all videos in dataset-good and dataset-bad
# This function processes all videos in the dataset-good and dataset-bad directories.
# It calls the process_video function for each video file in the directories.
# The extracted keypoints and squat features are saved to CSV files in the output directory.
# The output directory is created if it does not exist.
def process_all_videos():
    """Process all videos in dataset-good and dataset-bad"""
    for category in ["dataset-good", "dataset-bad"]:
        input_dir = os.path.join(RAW_DATA_DIR, category)
        output_dir = os.path.join(OUTPUT_DIR, category)

        os.makedirs(output_dir, exist_ok=True)

        for video_file in sorted(os.listdir(input_dir)):  # Sort for chronological order
            if video_file.endswith(".mov"):
                video_path = os.path.join(input_dir, video_file)
                output_csv = os.path.join(output_dir, video_file.replace(".mov", ".csv"))
                process_video_raw_features(video_path, output_csv)


if __name__ == "__main__":
    process_all_videos()

