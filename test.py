import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from mediapipe.tasks.python import BaseOptions
from mediapipe.framework.formats import landmark_pb2

# Define the model path
model_path = "pose_landmarker_full.task"

# Initialize MediaPipe Pose with video mode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO
)

def draw_landmarks(image, pose_landmarks, color):
    for landmark in pose_landmarks:
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 5, color, -1)

def to_normalized_landmark_list(landmarks):
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for landmark in landmarks:
        normalized_landmark = landmark_pb2.NormalizedLandmark(
            x=landmark.x, y=landmark.y, z=landmark.z, visibility=landmark.visibility)
        landmark_list.landmark.append(normalized_landmark)
    return landmark_list

def get_centroid(pose_landmarks):
    x_coords = [landmark.x for landmark in pose_landmarks]
    y_coords = [landmark.y for landmark in pose_landmarks]
    return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

# Initialize video capture
cap = cv2.VideoCapture('testing1.mp4')

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect poses
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = landmarker.detect_for_video(mp_image, timestamp_ms=int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        # Convert the image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks
        if results.pose_landmarks:
            for pose_landmarks in results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, to_normalized_landmark_list(pose_landmarks), mp.solutions.pose.POSE_CONNECTIONS)

        # Display the output
        cv2.imshow('Pose Tracking', image)

        # Exit when 'ESC' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()