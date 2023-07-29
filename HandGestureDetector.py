import cv2
import os
import HandDetector as hdm

cap = cv2.VideoCapture(0)
detector = hdm.handTracker()
gesture_threshold = 300

while True:
    success, img = cap.read()
    img = detector.handsFinder(img)
    lmlist = detector.positionFinder(img, draw=False)

    img = cv2.flip(img, 1)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break