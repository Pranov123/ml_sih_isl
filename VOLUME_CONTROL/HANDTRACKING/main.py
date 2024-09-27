import cv2
import handtracking_module as htm
import time

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

while True:
    success, image = cap.read()
    image = detector.findHands(image)
    lmList = detector.findPosition(image)
    if len(lmList) != 0:
        print(lmList[8])

    # Logic to calculate the FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Displaying FPS
    cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    cv2.imshow('Video', image)
    cv2.waitKey(1)
