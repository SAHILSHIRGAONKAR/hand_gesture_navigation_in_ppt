import cv2
import HandDetector as hdm

cap = cv2.VideoCapture(0)
detector = hdm.handTracker(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]


class FingersUpCounter:
    def __init__(self, handsNo):
        self.handsNo = handsNo

    def count_fingers(self, lmlist):
        if len(lmlist) != 0:
            fingers = []
            # thumb
            if lmlist[tipIds[0]][1] > lmlist[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # fingers
            for id in range(1, 5):
                if lmlist[tipIds[id]][2] < lmlist[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            totalfingers = fingers.count(1)
            return totalfingers
        else:
            return 0


def main():
    fingersUp = FingersUpCounter(handsNo=1)

    while True:
        success, img = cap.read()
        img = detector.handsFinder(img)
        lmlist = detector.positionFinder(img, draw=False)

        totalfingers = fingersUp.count_fingers(lmlist)
        print(totalfingers)

        img = cv2.flip(img, 1)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
