import cv2
import mediapipe as mp

"""The 21 hand landmarks.
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20
"""


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5):

        # Parameters for the mediapipe module
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackingCon)

        # To draw the hand landmarks (index,middle,thumb ...) and connect them.
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    # Returns the image with landmarks connected
    def findHands(self, image, draw=True):
        # Convert the BGR to RGB because the mediapipe uses only RGB
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # To check whether a hand is detected in the image.
        if self.results.multi_hand_landmarks:
            # The loop is to draw the landmarks/points for 'n' hands.
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)

        return image

    # Returns a list with coordinates of each landmark
    def findPosition(self, image, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                # Converting the coordinates/landmarks to pixels by multiplying it with width and height
                cx, cy = int(lm.x * w), int(lm.y * h)
                # id: gives us the landmark id (eg, 0: wrist,1 : thumb), lm : gives us the coordinates of each landmark
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (int(cx), int(cy)), 20, (0, 255, 0), cv2.FILLED)
        return lmList
