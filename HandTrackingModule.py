import time
import cv2
import mediapipe as mp
import numpy as np
import math


class handDectector():
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.handStats = {}
        self.mode = static_image_mode
        self.maxHands = max_num_hands
        self.detection_confidence = min_detection_confidence
        self.tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, drawMarks=True, duplicateHand=False):
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            height, width, channel = img.shape
            blank_image = np.zeros((height, width, channel), np.uint8)
            for handLm in self.results.multi_hand_landmarks:
                if drawMarks:
                    self.mpDraw.draw_landmarks(img, handLm, self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(121, 22, 76), thickness=2,
                                                                       circle_radius=4),
                                               self.mpDraw.DrawingSpec(color=(250, 44, 250), thickness=2,
                                                                       circle_radius=2), )
                if duplicateHand:
                    self.mpDraw.draw_landmarks(blank_image, handLm, self.mpHands.HAND_CONNECTIONS)
                    # cv2.imshow('Skeleton', blank_image)
            # if duplicateHand:
            #     resizeH, resizeW = 200, 150
            #     blank_image = cv2.resize(blank_image, (resizeW, resizeH))
            #     img[0:resizeH, 0:resizeW] = blank_image
        return img

    def findPositions(self, img, handNum=1, showId=True, showCircles=False):
        self.handStats = {}
        lmList = {}
        if self.results.multi_hand_landmarks:
            for hands in range(0, handNum):
                if hands >= len(self.results.multi_hand_landmarks):
                    continue
                hand = self.results.multi_hand_landmarks[hands]
                for id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList[id + (hands * 21)] = {'x': cx, 'y': cy}
                    if showId:
                        cv2.putText(img, str(id), (cx + 5, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 20, 220), 1)
                    if showCircles:
                        cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
                if len(lmList) == 21:
                    self.handStats[self.results.multi_handedness[0].classification[0].label] = lmList
                elif len(lmList) > 21:
                    self.handStats[self.results.multi_handedness[0].classification[0].label] = {i: lmList[i] for i in
                                                                                                range(0, 21)}
                    self.handStats[self.results.multi_handedness[1].classification[0].label] = {i - 21: lmList[i] for i
                                                                                                in
                                                                                                range(21, len(lmList))}
                if 'Left' in self.handStats:
                    left = (self.handStats['Left'][self.mpHands.HandLandmark.WRIST]['x'] + 10,
                            self.handStats['Left'][self.mpHands.HandLandmark.WRIST]['y'] + 10)
                    # cv2.putText(img, 'left', left, cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 20, 220), 2)
                if 'Right' in self.handStats:
                    right = (self.handStats['Right'][self.mpHands.HandLandmark.WRIST]['x'] + 10,
                             self.handStats['Right'][self.mpHands.HandLandmark.WRIST]['y'] + 10)
                    # cv2.putText(img, 'right', right, cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 20, 220), 2)
        return self.handStats

    def fingersOpen(self):
        fingers = {
            'thumb': {
                'id': self.mpHands.HandLandmark.THUMB_TIP
            },
            'index': {
                'id': self.mpHands.HandLandmark.INDEX_FINGER_TIP
            },
            'middle': {
                'id': self.mpHands.HandLandmark.MIDDLE_FINGER_TIP
            },
            'ring': {
                'id': self.mpHands.HandLandmark.RING_FINGER_TIP
            },
            'pinky': {
                'id': self.mpHands.HandLandmark.PINKY_TIP
            }
        }
        countFingers = {'Left': [], 'Right': []}
        if len(self.handStats) != 0:
            # print(lmList)
            for hand in self.handStats:
                fingerOpened = []
                for finger in fingers:
                    fingerTip = fingers[finger]['id']
                    if finger == 'thumb':
                        handInverted = self.handStats[hand][8]['x'] > self.handStats[hand][16]['x']
                        # print('handInverted', handInverted)
                        rightCheck = self.handStats[hand][fingerTip]['x'] < \
                                     self.handStats[hand][2]['x']
                        # print('rightCheck', rightCheck)
                        leftCheck = self.handStats[hand][fingerTip]['x'] < \
                                    self.handStats[hand][2]['x']
                        # print('leftCheck', leftCheck)

                        if (hand.lower() == 'right' and (
                                (not handInverted and rightCheck) ^ (handInverted and not rightCheck))) \
                                or \
                                (hand.lower() == 'left' and (
                                        (not handInverted and leftCheck) ^ (handInverted and not leftCheck))):
                            fingerOpened.append(finger)
                    elif self.handStats[hand][fingerTip]['y'] < self.handStats[hand][fingerTip - 2]['y']:
                        fingerOpened.append(finger)
                countFingers[hand] = fingerOpened
                # print("in mod countFingers = ", countFingers)
        return countFingers

    def findDistance(self, finger1, finger2, img, hand, draw=True, r=6, t=3):
        # print('in mod ==',self.handStats[hand])
        finger1X, finger1Y = self.handStats[hand][finger1]['x'], self.handStats[hand][finger1]['y']
        finger2X, finger2Y = self.handStats[hand][finger2]['x'], self.handStats[hand][finger2]['y']

        # print((finger1X, finger1Y),(finger2X, finger2Y))
        cx, cy = (finger1X + finger2X) // 2, (finger1Y + finger2Y) // 2

        if draw:
            cv2.line(img, (finger1X, finger1Y), (finger2X, finger2Y), (255, 0, 255), t)
            cv2.circle(img, (finger1X, finger1Y), r, (3, 3, 252), cv2.FILLED)
            cv2.circle(img, (finger2X, finger2Y), r, (3, 3, 252), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (31, 15, 255), cv2.FILLED)

        # length = math.hypot(finger2X-finger1X, finger2X - finger1Y)
        length = abs(finger2X - finger1X)
        # print('length',length,' == ',abs(finger2X-finger1X))

        plotPointsInfo = {
            'finger1X': finger1X,
            'finger1Y': finger1Y,
            'finger2X': finger2X,
            'finger2Y': finger1X,
            'centerX': cx,
            'centerY': cy
        }
        return length, img, plotPointsInfo


def main():
    cap = cv2.VideoCapture(0)
    prevTime = 0
    currTime = 0
    detector = handDectector()

    while True:
        success, img = cap.read()
        # lmList = {}
        #
        img = detector.findHands(img)
        lmList = detector.findPositions(img, handNum=1)

        if len(lmList) != 0:
            print(len(lmList))
            for i in lmList:
                print('result kaka = ', lmList[i][detector.mpHands.HandLandmark.WRIST])

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 20, 220), 3)
        #
        cv2.imshow('Video', img)

        char = cv2.waitKey(1) & 0xFF
        if char == 27 or chr(char) == 'q':
            break


if __name__ == '__main__':
    main()
