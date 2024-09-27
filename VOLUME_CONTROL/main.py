import cv2
import time
import numpy as np
from HANDTRACKING import handtracking_module as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize current and previous time for FPS calculation
pTime = 0
cTime = 0

# Access the speaker audio interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Initialize volume variables
vol = 0
volBar = 400
volPer = 0

# Set webcam dimensions
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize the hand detector
detector = htm.HandDetector(detectionCon=0.7)

while True:

    # Capture frame from webcam
    succ, img = cap.read()

    img = cv2.flip(img, 1)
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    # Only proceed when a hand is detected
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Centroid
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw circles on thumb tip, index finger tip, and center point
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        # Map the distance to volume range
        vol = np.interp(length, (40, 250), (minVol, maxVol))
        volume.SetMasterVolumeLevel(vol, None)

        if length < 40:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # Map the distance to volume bar and volume percentage
        volBar = np.interp(length, (40, 250), (400, 150))
        volPer = np.interp(length, (40, 250), (0, 100))

    # Draw the volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{str(int(fps))}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    cv2.putText(img, f'{str(int(volPer))}%', (55, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
