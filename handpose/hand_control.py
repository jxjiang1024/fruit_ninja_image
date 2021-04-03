'''
Source code for Fruit Ninja AI ( https://www.youtube.com/watch?v=Vw3vU9OdWAs ) 

The AI only loses when a bomb is overlapped with a fruit on its whole path, as the AI won't find a good opportunity to slice it.

The game as a chrome extension: https://chrome.google.com/webstore/detail/fruit-ninja-game/fdkhnibpmdfmpgaipjiodbpdngccfibp

Simply place the chrome extension on the top right corner of your screen and run this file :) 

Some heuristics and timings might differ depending on your machine.
(too much/little computing time between frames might affect the AI's decisions)
'''


import time
from datetime import datetime
import cv2
import numpy as np
import os
import win32api, win32con
import pyautogui
import sys
import threading
from time import sleep
import math
import keyboard
import cv2
import mediapipe as mp
from directkeys import PressKey, ReleaseKey, leftarrow, rightarrow, A, D, spacebar

DELAY_BETWEEN_SLICES = 0.01 # for sleep(DELAY_BETWEEN_SLICES)
DRAW_BOMBS = True
DEBUG = True
# pylint: disable=no-member,

screenHeight = win32api.GetSystemMetrics(1)
screenWidth = win32api.GetSystemMetrics(0)

print(screenHeight,screenWidth)

# actual screen coordinates : 1080,1920

def quit(): 
    exit()


'''
Moves mouse to (x,y) in screen coordinates
'''
def moveMouse(x, y):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, 
        int(x/screenWidth*65535.0), int(y/screenHeight*65535.0))





mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
coord_dict = []


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mouse_xcor,mouse_ycor = pyautogui.position()
    if results.multi_hand_landmarks:
        keypoint_str = str(results.multi_hand_landmarks)
        keypoint_list = keypoint_str.split('landmark ')
        necessary_keypoints =[]
        for i in range(len(keypoint_list)):
            keypoint_list[i] = keypoint_list[i].replace("{","")
            keypoint_list[i] = keypoint_list[i].replace("}","")
            keypoint_list[i] = keypoint_list[i].replace(" ","")
            keypoint_list[i] = keypoint_list[i].replace("\n","")
            if i in [5,9,13,17,21]:
                new_list = keypoint_list[i].split(":")
                necessary_keypoints.append([new_list[1][:-1],new_list[2][:-1]])
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            image_height, image_width, _ = image.shape


        thumb_xcoord = float(necessary_keypoints[0][0])
        thumb_ycoord = float(necessary_keypoints[0][1])

        index_xcoord = float(necessary_keypoints[1][0])
        index_ycoord = float(necessary_keypoints[1][1])

        middle_xcoord = float(necessary_keypoints[2][0])
        middle_ycoord = float(necessary_keypoints[2][1])

        ring_xcoord = float(necessary_keypoints[3][0])
        ring_ycoord = float(necessary_keypoints[3][1])

        pinky_xcoord = float(necessary_keypoints[4][0])
        pinky_ycoord = float(necessary_keypoints[4][1])


        average_peace_ycoord = float((thumb_ycoord + ring_ycoord + pinky_ycoord)/3)
        average_left_xcoord = float((middle_xcoord+ring_xcoord+pinky_xcoord)/3)
        diff_thumb_ring = abs(thumb_xcoord - ring_xcoord)
        diff_thumb_pinky = abs(thumb_xcoord - pinky_xcoord)


        if index_ycoord < average_peace_ycoord and middle_ycoord  < average_peace_ycoord and diff_thumb_ring<0.1*thumb_xcoord and diff_thumb_pinky<0.1*thumb_xcoord : 
            # PressKey(spacebar)
            print("space pressed")
        # pointing left
        elif thumb_xcoord > index_xcoord and index_xcoord < average_left_xcoord:
            # moveMouse(mouse_xcor*0.9,mouse_ycor)
            print("mouse moved, left")
        # pointing right
        elif thumb_xcoord < index_xcoord and index_xcoord > average_left_xcoord:
            # moveMouse(mouse_xcor*1.1,mouse_ycor)
            print("mouse moved, right")
            


    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        cv2.destroyAllWindows() 
        break
cap.release()

