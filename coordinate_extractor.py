import mediapipe as mp
import cv2
import numpy as np
import json

from utils.railway_dictionary import RAILWAY_IDS

class CoordinateExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
        )

    def extract_coordinates(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return []

        frame_landmarks = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(frame_rgb)
            hands_results = self.hands.process(frame_rgb)

            pose_landmarks = [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                }
                for lm in pose_results.pose_landmarks.landmark
            ] if pose_results.pose_landmarks else []

            hand_landmarks = [
                [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in hand_landmark.landmark
                ]
                for hand_landmark in hands_results.multi_hand_landmarks
            ] if hands_results.multi_hand_landmarks else []

            frame_landmarks.append({
                "pose": pose_landmarks,
                "hands": hand_landmarks,
            })

        cap.release()
        return frame_landmarks

def save_coordinates_to_file(coordinate_data, output_file="coordinates.py"):
    with open(output_file, "w") as f:
        f.write("COORDINATES = ")
        f.write(json.dumps(coordinate_data, indent=4))
        
def save_coordinates_to_json(coordinate_data, output_file="coordinates.json"):
    """
    Save extracted coordinate data to a JSON file.
    
    :param coordinate_data: Dictionary containing the landmarks for each word
    :param output_file: Path to the output JSON file
    """
    with open(output_file, "w") as f:
        json.dump(coordinate_data, f, indent=4)


if __name__ == "__main__":
    word_to_video_map = RAILWAY_IDS

    extractor = CoordinateExtractor()
    coordinates = {}

    for word, video_path in word_to_video_map.items():
        print(f"Processing word: {word}")
        coordinates[word] = extractor.extract_coordinates(video_path)

    save_coordinates_to_json(coordinates, output_file="coordinates.json")
    print("Coordinates saved to coordinates.json")
