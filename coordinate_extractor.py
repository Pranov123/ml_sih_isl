import mediapipe as mp
import cv2
import csv
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


def save_coordinates_to_csv(coordinate_data, output_file="coordinates.csv"):
    """
    Save coordinate data to a CSV file with word information.
    
    :param coordinate_data: Dictionary containing landmarks for each word.
    :param output_file: Path to the output CSV file.
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["word", "frame", "pose", "hands"]
        )
        writer.writeheader()

        for word, frames in coordinate_data.items():
            for frame_idx, frame_data in enumerate(frames):
                writer.writerow({
                    "word": word,
                    "frame": frame_idx,
                    "pose": json.dumps(frame_data.get("pose", [])),  # Convert list to JSON string
                    "hands": json.dumps(frame_data.get("hands", [])),  # Convert list to JSON string
                })

    print(f"Coordinates saved to {output_file}")



if __name__ == "__main__":
    word_to_video_map = RAILWAY_IDS # Mapping of words to video paths

    extractor = CoordinateExtractor()
    all_coordinates = {}  # Initialize an empty dictionary

    # Extract coordinates for each word
    for word, video_path in word_to_video_map.items():
        print(f"Processing word: {word}")
        coordinates = extractor.extract_coordinates(video_path)
        all_coordinates[word] = coordinates  # Associate the word with its coordinates

    # Save to CSV
    save_coordinates_to_csv(all_coordinates, output_file="coordinates.csv")

