import time
import cv2
import mediapipe as mp
import HandTrackingModule as htm
#
# cap = cv2.VideoCapture(0)
# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
#
# prevTime = 0
# currTime = 0
#
# while(True):
#     success,img = cap.read()
#
#     imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     results = hands.process(imgrgb)
#
#     if(results.multi_hand_landmarks):
#         for handLm in results.multi_hand_landmarks:
#             for id,lm in enumerate(handLm.landmark):
#                 h,w,c = img.shape
#                 cx,cy = int(lm.x*w),int(lm.y*h)
#                 cv2.putText(img, str(id), (cx+5, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 20, 220), 1)
#
#             mpDraw.draw_landmarks(img,handLm,mpHands.HAND_CONNECTIONS)
#
#     currTime = time.time()
#     fps = 1/(currTime-prevTime)
#     prevTime = currTime
#
#     cv2.putText(img,str(round(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(60, 20, 220),3)
#
#     cv2.imshow('Video',img)
#     print(results.multi_hand_landmarks)
#     cv2.waitKey(1)

cap = cv2.VideoCapture(0)
prevTime = 0
currTime = 0
detector = htm.handDectector()

while (True):
    success, img = cap.read()
    # lmList = []
    #
    # img = detector.findHands(img)
    # lmList = detector.findPositions(img,handNum=2,showCircles=True,showId = False)
    #
    # point = 0
    # if(len(lmList) != 0):
    #     print(lmList[point])
    #
    # currTime = time.time()
    # fps = 1 / (currTime - prevTime)
    # prevTime = currTime
    # cv2.putText(img, str(round(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 20, 220), 3)

    cv2.imshow('Video', img)

    char = cv2.waitKey(1) & 0xFF
    if(char == 27 or chr(char) == 'q'):
        break