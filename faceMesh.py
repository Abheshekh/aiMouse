import time
import cv2
import mediapipe as mp
import numpy as np
import math


class faceDetector():
    def __init__(self,
                 static_image_mode=True,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.faceStats = {}
        self.mode = static_image_mode
        self.maxFaces = max_num_faces
        self.detection_confidence = min_detection_confidence
        self.tracking_confidence = min_tracking_confidence

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.maxFaces, self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaces(self, img, drawMarks=False, skeletonView=False, showId=False):
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        self.faceStats = {}
        faceCount = 1
        if self.results.multi_face_landmarks:
            height, width, channel = img.shape
            blank_image = np.zeros((height, width, channel), np.uint8)
            for faceLMS in self.results.multi_face_landmarks:
                lmList = {}
                faceText = 'face'+str(faceCount)
                faceCount += 1
                if drawMarks:
                    self.mpDraw.draw_landmarks(img, faceLMS, self.mpFaceMesh.FACE_CONNECTIONS,
                                               self.drawSpec, self.drawSpec)

                if skeletonView:
                    self.mpDraw.draw_landmarks(blank_image, faceLMS, self.mpFaceMesh.FACE_CONNECTIONS,
                                               self.drawSpec, self.drawSpec)

                for faceLMId, lm in enumerate(faceLMS.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList[faceLMId] = {'x': cx, 'y': cy}
                    if showId:
                        cv2.putText(img, str(faceLMId), (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.18, (60, 20, 220), 1)
                        # if 33 >= faceLMId > 21:
                        #     cv2.putText(blank_image, str(faceLMId), (cx, cy),
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 20, 220), 1)
                self.faceStats[faceText] = lmList

                if skeletonView:
                    # self.mpDraw.draw_landmarks(blank_image, faceLMS, self.mpFaceMesh.FACE_CONNECTIONS,
                    #                            self.drawSpec, self.drawSpec)
                    resizeH, resizeW = 720, 1080
                    blank_image = cv2.resize(blank_image, (resizeW, resizeH))
                    cv2.imshow('Skeleton', blank_image)

            # if duplicateHand:
            #     resizeH, resizeW = 200, 150
            #     blank_image = cv2.resize(blank_image, (resizeW, resizeH))
            #     img[0:resizeH, 0:resizeW] = blank_image
        return img


def main():
    cap = cv2.VideoCapture(0)
    prevTime = 0
    currTime = 0
    detector = faceDetector()

    while True:
        success, img = cap.read()
        # img = cv2.imread('face.jpg')
        # lmList = {}
        #
        img = detector.findFaces(img, skeletonView=True)

        # if len(detector.faceStats) != 0:
        #     # print(len(detector.faceStats))
        #     for i in detector.faceStats:
        #         print('result kaka = ', detector.faceStats[i][0])

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
