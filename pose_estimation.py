import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mpPose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        if results.pose_landmarks:
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            h, w, c = frame.shape

            left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
            right_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            left_hip = (int(landmarks[23].x * w), int(landmarks[23].y * h))
            right_hip = (int(landmarks[24].x * w), int(landmarks[24].y * h))

            print(left_shoulder, right_shoulder)
        cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
