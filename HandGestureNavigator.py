import os
import cv2
from HandDetector import handTracker
# width , height = 1280, 720
FolderPath = "presentation"

cap = cv2.VideoCapture(0)
# cap.set(3, width)
# cap.set(4, height)
#get the list of presentation images
#sorted based on the length/numbers
pathImages = sorted(os.listdir(FolderPath), key=len)
# print(pathImages)

#variables
imgNumber = 0
hs, ws = 120, 230
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


#hand detector

detector = handTracker()


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(FolderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    imgCurrent = cv2.resize(imgCurrent, (1080, 620))

    img = detector.handsFinder(img)
    lm_list = detector.positionFinder(img)
    # print(lm_list)
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
        label_text = detector.results.multi_handedness[0].classification[
            0].label  # Replace this with the desired label text
        label_x = x_min
        label_y = y_max + 40  # Adjust the value (20) to set the distance between the square and the label

        # Draw the label on the image
        cv2.putText(img, label_text, (label_x, label_y), cv2.FONT_HERSHEY_COMPLEX,
                    FONT_SIZE, (0, 0, 255), FONT_THICKNESS, cv2.LINE_AA)


    #Adding webcam image on slide
    imgSmall = cv2.resize(img,(ws,hs))
    height, width, _ = imgCurrent.shape
    imgCurrent[0:hs,width-ws:width] = imgSmall



# return imgCurrent

    cv2.imshow("Image", img) #camera feed
    cv2.imshow("Slides", imgCurrent) #presentation feed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

