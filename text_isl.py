import asyncio
import cv2
import numpy as np
import mediapipe as mp
import torch
import tensorflow as tf
import time

from utils.drive_link_placeholder import DRIVE_LINK_PLACEHOLDER
from utils.connections import CONNECTIONS_NOT_NEEDED
from utils.idselector import VIDEO_ID

class GPULandmarkDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = tf.test.is_built_with_cuda()
        
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )

    def preprocess_for_gpu(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb.astype(np.uint8)

    def draw_landmarks(self, canvas, pose_landmarks, hand_landmarks, face_mesh_landmarks):
        pose_color = (0, 255, 0)  # Green
        hand_color = (255, 0, 0)  # Blue 
        face_color = (0, 0, 255)  # Red

        if pose_landmarks:
            for connection in self.mp_pose.POSE_CONNECTIONS:
                if connection[0] in CONNECTIONS_NOT_NEEDED or connection[1] in CONNECTIONS_NOT_NEEDED:
                    continue

                start_x, start_y = int(pose_landmarks.landmark[connection[0]].x * canvas.shape[1]), int(pose_landmarks.landmark[connection[0]].y * canvas.shape[0])
                end_x, end_y = int(pose_landmarks.landmark[connection[1]].x * canvas.shape[1]), int(pose_landmarks.landmark[connection[1]].y * canvas.shape[0])
                cv2.line(canvas, (start_x, start_y), (end_x, end_y), pose_color, 2)

        if hand_landmarks:
            for hand_landmark in hand_landmarks:
                for id, lm in enumerate(hand_landmark.landmark):
                    h, w, c = canvas.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(canvas, (cx, cy), 2, hand_color, cv2.FILLED)

                for connection in self.mp_hands.HAND_CONNECTIONS:
                    start_x, start_y = int(hand_landmark.landmark[connection[0]].x * w), int(hand_landmark.landmark[connection[0]].y * h)
                    end_x, end_y = int(hand_landmark.landmark[connection[1]].x * w), int(hand_landmark.landmark[connection[1]].y * h)
                    cv2.line(canvas, (start_x, start_y), (end_x, end_y), hand_color, 1)

        if face_mesh_landmarks:
            for face_landmarks in face_mesh_landmarks:
                for id, lm in enumerate(face_landmarks.landmark):
                    h, w, c = canvas.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(canvas, (cx, cy), 1, face_color, cv2.FILLED)

                for connection in self.mp_face_mesh.FACEMESH_CONTOURS:
                    start_x, start_y = int(face_landmarks.landmark[connection[0]].x * w), int(face_landmarks.landmark[connection[0]].y * h)
                    end_x, end_y = int(face_landmarks.landmark[connection[1]].x * w), int(face_landmarks.landmark[connection[1]].y * h)
                    cv2.line(canvas, (start_x, start_y), (end_x, end_y), face_color, 1)

        return canvas

    def detect_landmarks(self, image):
        image_rgb = self.preprocess_for_gpu(image)
        pose_results = self.pose.process(image_rgb)
        hands_results = self.hands.process(image_rgb)
        face_mesh_results = self.face_mesh.process(image_rgb)

        # White canvas
        canvas = 255 * np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        canvas = self.draw_landmarks(canvas, pose_results.pose_landmarks, hands_results.multi_hand_landmarks, face_mesh_results.multi_face_landmarks)

        return canvas

async def play_video(detector, video_path):
    """Plays a video and applies landmark detection, displaying FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    prev_time = 0  # To calculate the time between frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (500, 500))
        landmark_canvas = detector.detect_landmarks(frame)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Display FPS on the video frame
        cv2.putText(
            landmark_canvas, 
            f"FPS: {int(fps)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            1, 
            cv2.LINE_AA
        )

        cv2.imshow("Landmark Canvas", landmark_canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

async def fetch_and_play_video(detector, word, word_to_video_map):
    """Fetch and play a video for a word asynchronously."""
    video_path = word_to_video_map.get(word.lower())
    if video_path:
        print(f"Playing video for word: {word}")
        await play_video(detector, video_path)
    else:
        print(f"No video mapped for word: {word}")
        await asyncio.sleep(1)  # Add a small delay for unmapped words

async def process_sentence(sentence, word_to_video_map):
    """Processes a sentence and plays corresponding videos concurrently."""
    detector = GPULandmarkDetector()

    print(f"GPU Available: {detector.gpu_available}")
    print(f"Using device: {detector.device}")

    words = sentence.split()

    # Create tasks to fetch and play videos concurrently
    tasks = [fetch_and_play_video(detector, word, word_to_video_map) for word in words]
    
    # Run tasks concurrently
    await asyncio.gather(*tasks)

# Usage
if __name__ == "__main__":
    sentence = "what your name"
    # Run the process asynchronously
    asyncio.run(process_sentence(sentence, VIDEO_ID))
