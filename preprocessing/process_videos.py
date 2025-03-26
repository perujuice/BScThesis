import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Paths to datasets
RAW_DATA_DIR = "assets/raw_data"
OUTPUT_DIR = "assets/extracted_keypoints"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Landmarks indices
LANDMARKS = {
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_shoulder": 11, "right_shoulder": 12
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
    """Calculate angle between three points (shoulder, hip, ankle)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Process video and extract keypoints
# video_path is the path to the video file
# output_csv is the path to the output CSV file
# This function processes a video file frame by frame using MediaPipe Pose.
# It extracts keypoints from the pose landmarks and calculates squat features.
# The features extracted are knee valgus ratio, torso lean angle, and squat depth.
# The knee valgus ratio is the ratio of the distance between the knees to the distance between the hips.
# The torso lean angle is the angle between the shoulder, hip, and ankle.
# The squat depth is the distance between the hip and knee.
# The extracted features are saved to a CSV file.
def process_video(video_path, output_csv):
    """Extract keypoints and calculate squat features"""
    cap = cv2.VideoCapture(video_path)
    data = []
    frame_count = 0

    # Process each frame
    # This loop processes each frame of the video and extracts keypoints using MediaPipe Pose
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            frame_data = {"frame": frame_count}

            # Get landmark positions
            lm = results.pose_landmarks.landmark

            '''
            For the knee valgus logic, I referred to the following paper:
            "The back squat: A proposed assessment of functional deficits and technical factors that limit performance"
            https://pmc.ncbi.nlm.nih.gov/articles/PMC4262933/#S13
            '''

            # Compute Knee Valgus (Frontal Plane Knee Angle) per leg
            hip_left = (lm[LANDMARKS["left_hip"]].x, lm[LANDMARKS["left_hip"]].y)
            knee_left = (lm[LANDMARKS["left_knee"]].x, lm[LANDMARKS["left_knee"]].y)
            ankle_left = (lm[LANDMARKS["left_ankle"]].x, lm[LANDMARKS["left_ankle"]].y)
            valgus_angle_left = calculate_angle(hip_left, knee_left, ankle_left)

            hip_right = (lm[LANDMARKS["right_hip"]].x, lm[LANDMARKS["right_hip"]].y)
            knee_right = (lm[LANDMARKS["right_knee"]].x, lm[LANDMARKS["right_knee"]].y)
            ankle_right = (lm[LANDMARKS["right_ankle"]].x, lm[LANDMARKS["right_ankle"]].y)
            valgus_angle_right = calculate_angle(hip_right, knee_right, ankle_right)

            # Compute Torso Lean Angle
            shoulder = (lm[LANDMARKS["left_shoulder"]].x, lm[LANDMARKS["left_shoulder"]].y)
            hip = (lm[LANDMARKS["left_hip"]].x, lm[LANDMARKS["left_hip"]].y)
            ankle = (lm[LANDMARKS["left_ankle"]].x, lm[LANDMARKS["left_ankle"]].y)
            torso_angle = calculate_angle(shoulder, hip, ankle)

            # Compute Squat Depth (Hip below knee) 
            # If the hip is above the knee, the squat depth is negative
            # If the hip is below the knee, the squat depth is positive
            squat_depth = lm[LANDMARKS["left_hip"]].y - lm[LANDMARKS["left_knee"]].y

            # Save extracted features
            # Save both left and right knee valgus angles
            frame_data["valgus_angle_left"] = valgus_angle_left
            frame_data["valgus_angle_right"] = valgus_angle_right
            frame_data["torso_angle"] = torso_angle
            frame_data["squat_depth"] = squat_depth
            data.append(frame_data)

        frame_count += 1

    cap.release()

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Processed {video_path} -> {output_csv}")


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
                process_video(video_path, output_csv)



if __name__ == "__main__":
    process_all_videos()

