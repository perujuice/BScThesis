import cv2
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Landmarks indices
LANDMARKS = {
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_shoulder": 11, "right_shoulder": 12
}

def calculate_angle(a, b, c):
    """Calculate angle at point b given three 2D points (a-b-c)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def show_video_with_features(video_path):
    while True:  # Loop playback
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Failed to open video.")
            break

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video, exit inner loop and reopen

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Torso angle (shoulder–hip–ankle)
                shoulder = (lm[LANDMARKS["left_shoulder"]].x, lm[LANDMARKS["left_shoulder"]].y)
                hip = (lm[LANDMARKS["left_hip"]].x, lm[LANDMARKS["left_hip"]].y)
                ankle = (lm[LANDMARKS["left_ankle"]].x, lm[LANDMARKS["left_ankle"]].y)
                torso_angle = calculate_angle(shoulder, hip, ankle)

                # Knee valgus (hip–knee–ankle) angles
                hip_left = (lm[LANDMARKS["left_hip"]].x, lm[LANDMARKS["left_hip"]].y)
                knee_left = (lm[LANDMARKS["left_knee"]].x, lm[LANDMARKS["left_knee"]].y)
                ankle_left = (lm[LANDMARKS["left_ankle"]].x, lm[LANDMARKS["left_ankle"]].y)
                valgus_left = calculate_angle(hip_left, knee_left, ankle_left)

                hip_right = (lm[LANDMARKS["right_hip"]].x, lm[LANDMARKS["right_hip"]].y)
                knee_right = (lm[LANDMARKS["right_knee"]].x, lm[LANDMARKS["right_knee"]].y)
                ankle_right = (lm[LANDMARKS["right_ankle"]].x, lm[LANDMARKS["right_ankle"]].y)
                valgus_right = calculate_angle(hip_right, knee_right, ankle_right)

                squat_depth = lm[LANDMARKS["left_hip"]].y - lm[LANDMARKS["left_knee"]].y

                # Display overlays
                cv2.putText(frame, f"Valgus L: {valgus_left:.1f}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                cv2.putText(frame, f"Valgus R: {valgus_right:.1f}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                cv2.putText(frame, f"Torso Angle: {torso_angle:.1f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Squat Depth: {squat_depth:.3f}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)

            cv2.imshow("Squat Feature Visualizer (Press 'q' to quit)", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return  # Exit completely

        cap.release()  # Finished the video — go to next loop

# Run it
if __name__ == "__main__":
    video_path = "assets/raw_data/dataset-good/good1.mov"  # Change this if needed
    show_video_with_features(video_path)
