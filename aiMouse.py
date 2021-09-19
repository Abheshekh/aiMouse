import time
import cv2
import mediapipe as mp
import HandTrackingModule as htm
import os
import pyautogui
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
from datetime import datetime as d
import os

screenW, screenH = pyautogui.size()

camW, camH = 640, 480

escFlag = False
dragFlag = False
alt_tab_flag = False
win_tab_flag = False
brightens_flag = False
sound_flag = False
ss_flag = False
delay = 0


def set_volume_driver():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    return volume


class aiMouse:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.setCam()
        self.prevTime = 0
        self.currTime = 0
        self.prevLocationX = 0
        self.prevLocationY = 0
        self.smoothening = 8
        self.regionOfInterestX = (130, 450)
        self.regionOfInterestY = (30, 230)
        self.img = None
        self.volume = set_volume_driver()

    def setCam(self):
        self.cap.set(3, camW)
        self.cap.set(4, camH)


def main():
    mouse = aiMouse()
    global escFlag, camH, camW
    handDetector = htm.handDectector(min_detection_confidence=0.75)
    while True:
        success, mouse.img = mouse.cap.read()
        mouse.img = handDetector.findHands(mouse.img, duplicateHand=True)
        cv2.rectangle(mouse.img, (mouse.regionOfInterestX[0] + 10, mouse.regionOfInterestY[0] + 10),
                      (mouse.regionOfInterestX[1] - 10, mouse.regionOfInterestY[1] - 10),
                      (0, 0, 0), 1)
        cv2.rectangle(mouse.img, (mouse.regionOfInterestX[0], mouse.regionOfInterestY[0]),
                      (mouse.regionOfInterestX[1], mouse.regionOfInterestY[1]),
                      (0, 0, 0), 2)
        handStats = handDetector.findPositions(mouse.img, handNum=1)
        if len(handStats) != 0:
            hand = list(handStats.keys())[-1]
            lmList = handStats[hand]
            fingersOpen = handDetector.fingersOpen()[hand]
            gestures(fingersOpen, lmList, mouse, handDetector, hand)
        mouse.currTime = time.time()
        # print(time.time())
        fps = 1 / (mouse.currTime - mouse.prevTime)
        mouse.prevTime = mouse.currTime
        cv2.putText(mouse.img, 'FPS:' + str(round(fps)), (10, camH - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (60, 20, 220), 2)

        cv2.imshow('Video', mouse.img)
        char = cv2.waitKey(1) & 0xFF
        if escFlag:
            char = ord('q')
        if char == 27 or chr(char) == 'q':
            break


def gestures(fingersOpen, lmList, mouse, handDetector, hand):
    global escFlag
    global dragFlag
    global alt_tab_flag
    global win_tab_flag
    global brightens_flag
    global sound_flag
    global ss_flag, delay
    indexX, indexY = lmList[8]['x'], lmList[8]['y']
    if len(fingersOpen) == 5:
        if dragFlag:
            pyautogui.mouseUp(button='left')
            dragFlag = False
        if any([brightens_flag, sound_flag, ss_flag]):
            brightens_flag = False
            sound_flag = False
            ss_flag = False
    elif any([brightens_flag, sound_flag]):
        if brightens_flag:
            length, mouse.img, plotInfo = handDetector.findDistance(4, 8, mouse.img, hand)
            brightness = np.interp(length, [6, 80], [0, 100])
            sbc.set_brightness(brightness)
        elif sound_flag:
            length, mouse.img, plotInfo = handDetector.findDistance(4, 8, mouse.img, hand)
            volume_range = mouse.volume.GetVolumeRange()
            min_volume = volume_range[0]
            max_volume = volume_range[1]
            vol = np.interp(length, [6, 80], [min_volume, max_volume])
            # print(length, vol)
            mouse.volume.SetMasterVolumeLevel(vol, None)
    elif len(fingersOpen) == 0:
        if not dragFlag:
            pyautogui.mouseDown(button='left')
            dragFlag = True
    elif 'index' in fingersOpen and len(fingersOpen) == 1:
        move_pointer(indexX, indexY, mouse)
    elif sorted(set(['index', 'thumb'])) == sorted(set(fingersOpen)):
        length, img, plotInfo = handDetector.findDistance(4, 5, mouse.img, hand)
        if length < 44:
            cv2.circle(img, (plotInfo['centerX'], plotInfo['centerY']), 6,
                       (31, 255, 15), cv2.FILLED)
            pyautogui.click()
            if alt_tab_flag or win_tab_flag:
                alt_tab_flag = False
                win_tab_flag = False
                pyautogui.keyUp('alt')
                pyautogui.keyUp('win')
    elif sorted(set(['index', 'middle'])) == sorted(set(fingersOpen)):
        length, mouse.img, plotInfo = handDetector.findDistance(8, 12, mouse.img, hand)
        if length < 30:
            cv2.circle(mouse.img, (plotInfo['centerX'], plotInfo['centerY']), 6,
                       (31, 255, 15), cv2.FILLED)
            pyautogui.click(button='right')
    elif sorted(set(['index', 'pinky'])) == sorted(set(fingersOpen)):
        pyautogui.scroll(-10)
    elif sorted(set(['index', 'pinky', 'thumb'])) == sorted(set(fingersOpen)):
        pyautogui.scroll(10)
    elif sorted(set(['index', 'middle', 'ring'])) == sorted(set(fingersOpen)):
        if not alt_tab_flag:
            pyautogui.keyDown('alt')
            time.sleep(.2)
            pyautogui.press('tab')
        alt_tab_flag = True
    elif sorted(set(['index', 'middle', 'ring', 'pinky'])) == sorted(set(fingersOpen)):
        if not win_tab_flag:
            pyautogui.keyDown('win')
            time.sleep(.2)
            pyautogui.press('tab')
            pyautogui.keyUp('win')
        win_tab_flag = True
    elif sorted(set(['pinky', 'thumb', 'middle', 'ring'])) == sorted(set(fingersOpen)):
        if not escFlag:
            escFlag = True
    elif sorted(set(['index', 'middle', 'pinky'])) == sorted(set(fingersOpen)):
        sound_flag = True
    elif sorted(set(['index', 'ring', 'pinky'])) == sorted(set(fingersOpen)):
        brightens_flag = True
    elif sorted(set(['middle', 'thumb'])) == sorted(set(fingersOpen)):
        if delay != 3 and not ss_flag:
            delay += 1
            time.sleep(0.5)
        elif 'Screenshots' not in os.listdir():
            os.mkdir('Screenshots')
        elif not ss_flag:
            date = d.now()
            myScreenshot = pyautogui.screenshot()
            cwd = os.getcwd()
            filename = 'ScreenShot_py' + str(date) + '.png'
            filename = filename.replace(':', '-')
            path = cwd + '\\Screenshots\\' + filename
            print(str(date), path, cwd, filename, sep='\n')
            myScreenshot.save(path)
            ss_flag = True
            delay = 0


def move_pointer(indexX, indexY, mouse):
    pointerPositionX = np.interp(indexX, mouse.regionOfInterestX, (0, screenW))
    pointerPositionY = np.interp(indexY, mouse.regionOfInterestY, (0, screenH))

    currLocationX = mouse.prevLocationX + (pointerPositionX - mouse.prevLocationX) // mouse.smoothening
    currLocationY = mouse.prevLocationY + (pointerPositionY - mouse.prevLocationY) // mouse.smoothening

    pyautogui.moveTo(currLocationX, currLocationY)
    cv2.circle(mouse.img, (indexX, indexY), 6, (3, 173, 252), cv2.FILLED)

    mouse.prevLocationX = currLocationX
    mouse.prevLocationY = currLocationY


if __name__ == '__main__':
    main()
