import cv2
import mediapipe as mp
#lass encapsulates the hand tracking functionality and provides methods to detect hands in an image
# and find the positions of specific landmarks.
class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands  #An instance of the MediaPipe Hands class,
                                           # which provides hand tracking functionality.
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon) #The actual hand tracking object,
                                                                          # initialized with the specified parameters.

        self.mpDraw = mp.solutions.drawing_utils   #An instance of the MediaPipe Drawing utilities class,
                                                   # used to draw the landmarks on the detected hands


    #This method takes an image (frame from the video stream) as input
    #and returns the image with the detected hands and their landmarks drawn on it
    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    #img feed is in BGR format mediapipe requires
                                                             #to convert in RGB format
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                    # print('Handedness:', self.results.multi_handedness)
                    # Handedness = self.results.multi_handedness
                    # print(Handedness)
        return image


    #This method takes an image (frame from the video stream) as input,
    #and it returns a list of landmarks' positions for a specified hand.
    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            # if draw:
            #     cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmlist
# variables
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    while True:
        success, image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image, draw=False)
        if len(lmList) != 0:
            # Get the bounding box coordinates for the hand
            x_min, y_min = lmList[0][1], lmList[0][2]
            x_max, y_max = x_min, y_min
            for lm in lmList:
                x, y = lm[1], lm[2]
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Draw a square around the hand
            cv2.rectangle(image, (x_min - MARGIN, y_min - MARGIN), (x_max + MARGIN, y_max + MARGIN), (255, 0, 0), 2)

            # Calculate the position for the label
            label_text = tracker.results.multi_handedness[0].classification[0].label  # Replace this with the desired label text
            label_x = x_min
            label_y = y_max + 40  # Adjust the value (20) to set the distance between the square and the label

            # Draw the label on the image
            cv2.putText(image, label_text, (label_x, label_y), cv2.FONT_HERSHEY_COMPLEX,
                        FONT_SIZE, (0, 0, 255), FONT_THICKNESS, cv2.LINE_AA)

        cv2.imshow("Video", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
