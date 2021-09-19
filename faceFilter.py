import time
import cv2
import mediapipe as mp
import numpy as np
import math
import faceMesh as fM

prevTime = 0
currTime = 0
detector = fM.faceDetector()
cap = cv2.VideoCapture(0)
maskImg = cv2.imread('mask.png')
maskImg = cv2.flip(maskImg, 1)

while True:
    success, img = cap.read()

    img = detector.findFaces(img, skeletonView=True)

    if len(detector.faceStats) != 0:
        facePoints = detector.faceStats['face1']
        noseCenter = (facePoints[4]['x'], facePoints[4]['y'])
        cv2.circle(img, noseCenter, 8, (0, 255, 255), cv2.FILLED)
        topPoint = (facePoints[10]['x'], facePoints[10]['y'])
        cv2.circle(img, topPoint, 8, (0, 255, 255), cv2.FILLED)
        chinCenter = (facePoints[152]['x'], facePoints[152]['y'])
        cv2.circle(img, chinCenter, 10, (0, 255, 255), cv2.FILLED)
        leftCenter = (facePoints[234]['x'], facePoints[234]['y'])
        cv2.circle(img, leftCenter, 8, (0, 255, 255), cv2.FILLED)
        rightCenter = (facePoints[454]['x'], facePoints[454]['y'])
        cv2.circle(img, rightCenter, 8, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (facePoints[234]['x'], facePoints[10]['y']), 10, (0, 255, 255), cv2.FILLED)
        cv2.rectangle(img, (leftCenter[0], topPoint[1]),
                      (rightCenter[0], chinCenter[1]), (255, 0, 255), 5)

        maskWidth = int(math.hypot(leftCenter[1] - rightCenter[1], leftCenter[0] - rightCenter[0]))
        maskHeight = int(maskWidth * 1.3)
        maskImg1 = cv2.resize(maskImg, (maskWidth, maskHeight))
        maskImg1Gray = cv2.cvtColor(maskImg1, cv2.COLOR_BGR2GRAY)
        _,maskMask = cv2.threshold(maskImg1Gray, 25, 255, cv2.THRESH_BINARY_INV)

        maskArea = img[topPoint[1]:topPoint[1] + maskHeight, leftCenter[0]:leftCenter[0] + maskWidth]

        preFinalImg = cv2.bitwise_or(maskArea, maskArea, mask=maskMask)
        finalImg = cv2.add(preFinalImg, maskImg1)

        img[topPoint[1]:topPoint[1] + maskHeight, leftCenter[0]:leftCenter[0] + maskWidth] = finalImg

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 20, 220), 3)

    cv2.imshow('Video', img)
    cv2.imshow('finalImg', finalImg)

    char = cv2.waitKey(1) & 0xFF
    if char == 27 or chr(char) == 'q':
        break
