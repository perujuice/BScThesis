import cv2
import mediapipe as mp

# Load the video
video_path = "assets/raw_data/dataset-good/good1.mov"  # Change this to your video file
cap = cv2.VideoCapture(video_path)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect keypoints
    results = pose.process(rgb_frame)

    # Draw keypoints
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            h, w, _ = frame.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Squat Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
