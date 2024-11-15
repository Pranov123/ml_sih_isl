import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the models
single_hand_model_dict = pickle.load(open('saved_models/single_hand_model_word_seq(scikit-upgraded).p', 'rb'))
single_hand_model = single_hand_model_dict['model']

double_hand_model_dict = pickle.load(open('saved_models/double_hand_model_word(scikit-upgraded).p', 'rb'))
double_hand_model = double_hand_model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

# Labels for single and double hand signs
double_hand_labels_dict = {0: 'A', 1: 'B', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'J', 8: 'K', 9: 'M', 10: 'N',
                           11: 'P', 12: 'Q', 13: 'R', 14: 'S', 15: 'T', 16: 'W', 17: 'X', 18: 'Y', 19: 'Z',20:'ACCIDENT',21:'HELP'}

single_hand_labels_dict = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: 'C', 10: 'I',
                           11: 'L', 12: 'O', 13: 'U', 14: 'V',15:'PAIN',16:'CALL',17:'NEXT',18:'BACKSPACE',19:'SPACE'}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)  # Count the number of detected hands
        print(f"Number of hands detected: {num_hands}")

        # Draw landmarks for each hand
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmark coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Get bounding box coordinates
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Check if it's a single hand or double hand and predict accordingly
        try:
            if num_hands == 1:
                # Single hand prediction
                prediction = single_hand_model.predict([np.asarray(data_aux)])
                predicted_character = single_hand_labels_dict[int(prediction[0])]
            else:
                # Double hand prediction
                prediction = double_hand_model.predict([np.asarray(data_aux)])
                predicted_character = double_hand_labels_dict[int(prediction[0])]

            # Draw bounding box and predicted character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        except ValueError as e:
            print(f"Error during prediction: {e}")

    # Display the frame
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
