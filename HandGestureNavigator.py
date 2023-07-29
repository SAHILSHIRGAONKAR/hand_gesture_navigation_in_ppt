import os
import cv2
import time
from HandDetector import handTracker

wc, hc = 380, 480
FolderPath = "presentation"

cap = cv2.VideoCapture(0)
cap.set(3, wc)
cap.set(4, hc)

# get the list of presentation images
# sorted based on the length/numbers
pathImages = sorted(os.listdir(FolderPath), key=len)

# variables
imgNumber = 0
hs, ws = 120, 230
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
gesture_threshold = 130
tipIds = [4, 8, 12, 16, 20]
buffer_time = 0.8
# hand detector
detector = handTracker()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(FolderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    imgCurrent = cv2.resize(imgCurrent, (1080, 620))

    img = detector.handsFinder(img)
    cv2.line(img, (0, gesture_threshold), (wc, gesture_threshold), (0, 255, 0), 10)
    lm_list = detector.positionFinder(img)

    if len(lm_list) != 0:
        # Get the bounding box coordinates for the hand
        x_min, y_min = lm_list[0][1], lm_list[0][2]
        x_max, y_max = x_min, y_min
        for lm in lm_list:
            x, y = lm[1], lm[2]
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        # Draw a square around the hand
        cv2.rectangle(img, (x_min - MARGIN, y_min - MARGIN), (x_max + MARGIN, y_max + MARGIN), (255, 0, 0), 2)

        # Calculate the position for the label
        label_text = detector.results.multi_handedness[0].classification[0].label
        label_x = x_min
        label_y = y_max + 40  # Adjust the value (20) to set the distance between the square and the label

        # Draw the label on the image
        cv2.putText(img, label_text, (label_x, label_y), cv2.FONT_HERSHEY_COMPLEX,
                    FONT_SIZE, (0, 0, 255), FONT_THICKNESS, cv2.LINE_AA)

        # Finger counting
        lmlist = detector.positionFinder(img, draw=False)
        # print(lmlist)

        if len(lmlist) != 0:
            fingers = []
            # thumb
            if lmlist[tipIds[0]][1] < lmlist[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # fingers
            for id in range(1, 5):
                if lmlist[tipIds[id]][2] < lmlist[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        # print(fingers)  # Print the list that fingers returns

        # Check if the center of the hand is above the gesture threshold
        center_y = (y_min + y_max) // 2
        if center_y <= gesture_threshold:
            indexFinger = lmlist[8],[0], lmlist[8][1]
            #gesture no1 = left
            if fingers == [1,0,0,0,0]:
                print("left")
                if imgNumber > 0:
                    imgNumber-=1
                    time.sleep(buffer_time)
            # gesture no2 = Right
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                if imgNumber < len(pathImages)-1:
                    imgNumber+=1
                    time.sleep(buffer_time)

            #gesture 3 show Pointer
            if fingers == [0, 1, 0, 0, 0]:
                cv2.circle(imgCurrent, indexFinger, 12,(0,0,255),cv2.FILLED)
                    # time.sleep(buffer_time)

    # Adding webcam image on slide
    imgSmall = cv2.resize(img, (ws, hs))
    height, width, _ = imgCurrent.shape
    imgCurrent[0:hs, width - ws:width] = imgSmall

    cv2.imshow("Image", img)  # camera feed
    cv2.imshow("Slides", imgCurrent)  # presentation feed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
