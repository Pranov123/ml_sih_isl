import cv2
import numpy as np
import mediapipe as mp
import torch
import tensorflow as tf

class GPULandmarkDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = tf.test.is_built_with_cuda()

        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def preprocess_for_gpu(self, image):
        # Convert image to RGB and uint8
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb.astype(np.uint8)

    def detect_landmarks(self, image):
        image_rgb = self.preprocess_for_gpu(image)
        pose_results = self.pose.process(image_rgb)
        hands_results = self.hands.process(image_rgb)

        image_with_landmarks = image.copy()
        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image_with_landmarks,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image_with_landmarks,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return image_with_landmarks

def process_video(video_url):
    detector = GPULandmarkDetector()
    
    print(f"GPU Available: {detector.gpu_available}")
    print(f"Using device: {detector.device}")
    
    cap = cv2.VideoCapture(video_url)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (400, 400))
        frame_with_landmarks = detector.detect_landmarks(frame)
        
        cv2.imshow("GPU-Accelerated Landmark Detection", frame_with_landmarks)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# Usage
if __name__ == "__main__":
    file_id = "1MCKTmGVAJn_y3pc6PN0j9vJ6uHZIwNm6"
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    process_video(direct_url)