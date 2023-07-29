import cv2
import os
import HandDetector as hdm
class HandDetector:
    def __init__(self, detectionCon=0.75):
        self.cap = cv2.VideoCapture(0)
        self.detector = hdm.handTracker(detectionCon=0.75)
        self.tipIds = [4, 8, 12, 16, 20]

    def FingersUpCounter(self):
        while True:
            success, img = self.cap.read()
            img = self.detector.handsFinder(img)
            lmlist = self.detector.positionFinder(img, draw=False)

            if len(lmlist) != 0:
                fingers = []
                # Thumb
                if lmlist[self.tipIds[0]][1] > lmlist[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # Fingers
                for id in range(1, 5):
                    if lmlist[self.tipIds[id]][2] < lmlist[self.tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                print(fingers)

            img = cv2.flip(img, 1)
            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_detector = HandDetector(detectionCon=0.75)
    hand_detector.FingersUpCounter()
