import time
import cv2
import mediapipe as mp
import HandTrackingModule as htm
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)

wcam, hcam = 640, 480
cap.set(3, wcam)
cap.set(4, hcam)

prevTime = 0
currTime = 0

handDetector = htm.handDectector(min_detection_confidence=0.75)

fingers = {
    'thumb': {
        'position': handDetector.mpHands.HandLandmark.THUMB_TIP,
        'opened': False
    },
    'index': {
        'position': handDetector.mpHands.HandLandmark.INDEX_FINGER_TIP,
        'opened': False
    },
    'middle': {
        'position': handDetector.mpHands.HandLandmark.MIDDLE_FINGER_TIP,
        'opened': False
    },
    'ring': {
        'position': handDetector.mpHands.HandLandmark.RING_FINGER_TIP,
        'opened': False
    },
    'pinky': {
        'position': handDetector.mpHands.HandLandmark.PINKY_TIP,
        'opened': False
    }
}

while (True):
    success, img = cap.read()

    img = handDetector.findHands(img, duplicateHand=True)
    lmList = handDetector.findPositions(img, handNum=2)

    # if (len(lmList) != 0):
    #     for i in lmList:
    #         print('result kaka = ', lmList[i][handDetector.mpHands.HandLandmark.WRIST])
    if (len(lmList) != 0):
        # print(lmList)
        countFingers = {'left': 0, 'right': 0}
        for hand in lmList:
            fingerCount = 0
            for finger in fingers:
                fingerPosition = fingers[finger]['position']
                if (finger == 'thumb'):
                    handInverted = lmList[hand][8]['x'] > lmList[hand][16]['x']
                    print('handInverted',handInverted)
                    rightCheck = lmList[hand][fingerPosition]['x'] < lmList[hand][fingerPosition - 1]['x']
                    print('rightCheck',rightCheck)
                    leftCheck = lmList[hand][fingerPosition]['x'] < lmList[hand][fingerPosition - 1]['x']
                    print('leftCheck',leftCheck)

                    if (hand.lower() == 'right' and ((not handInverted and rightCheck) ^ (handInverted and not rightCheck))) \
                            or \
                        (hand.lower() == 'left' and ((not handInverted and leftCheck) ^ (handInverted and not leftCheck))):
                        fingerCount += 1
                        fingers[finger]['opened'] = True
                    else:
                        fingers[finger]['opened'] = False
                elif (lmList[hand][fingerPosition]['y'] < lmList[hand][fingerPosition - 2]['y']):
                    fingerCount += 1
                    fingers[finger]['opened'] = True
                else:
                    fingers[finger]['opened'] = False

            countFingers[hand.lower()] = fingerCount
        cv2.putText(img, 'Left:' + str(countFingers['left']), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 20, 220), 1)
        cv2.putText(img, 'Right:' + str(countFingers['right']), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 20, 220),
                    1)
        cv2.putText(img, 'Total:' + str(countFingers['left'] + countFingers['right']), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 20, 220), 1)
        # overlayH,overlayW,overlayC = overlayImgList[fingerCount].shape
        # img[0:overlayH,0:overlayW] = overlayImgList[fingerCount]

    imgH, imgW, imgC = img.shape
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, 'FPS:' + str(round(fps)), (10, imgH - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 20, 220), 2)

    cv2.imshow('Video', img)
    char = cv2.waitKey(1) & 0xFF
    if (char == 27 or chr(char) == 'q'):
        break
