import time
from datetime import datetime
import cv2
import mss
import numpy as np
import os
# import win32api, win32con
import pyautogui
import sys
import threading
from time import sleep
import math
# import keyboard
time.sleep(5)

# pylint: disable=no-member,

# screenHeight = win32api.GetSystemMetrics(1)
# screenWidth = win32api.GetSystemMetrics(0)
screenWidth, screenHeight = pyautogui.size()
'''
The game resolution is 750x500
'''
# width = 750
# height = 500
# multiply by 0.6

width = 1800
height = 2000

'''
I'm displaying my game at the top right corner of my screen
'''
gameScreen = {'top': 150, 'left': screenWidth - width, 'width': width, 'height': height}
# gameScreen = {'top': 200, 'left': 600, 'width': width, 'height': height}


today = datetime.now()

timeString = today.strftime('%b_%d_%Y__%H_%M_%S')

# Set writeVideo to True for saving screen captures for youtube
writeVideo = ( len(sys.argv) > 1 and sys.argv[1] == 'save' ) 

if(writeVideo):
    outputDir = './out/' + timeString
    os.mkdir(outputDir)
    fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
    outScreen = cv2.VideoWriter(outputDir + '/screen.avi', fourcc1, 25, (width, height))

    fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
    outResult = cv2.VideoWriter(outputDir + '/result.avi', fourcc2, 25, (width, height))

    fourcc3 = cv2.VideoWriter_fourcc(*'XVID')
    outMask = cv2.VideoWriter(outputDir + '/mask.avi', fourcc3, 25, (width, height))

def quit(): 
        
    if(writeVideo):
        outScreen.release()
        outResult.release()
        outMask.release()
    exit()


'''
Check if margins match screen coordinates
'''
def sanitizeMargins(rx, ry):
    margin = 10
    if (rx > width - margin):
        rx = width - margin
    if (rx < margin):
        rx = margin
    if ry > height - margin:
        ry = height - margin
    if ry < margin:
        ry = margin
    return (rx, ry)

'''
Translates from game screen coordinates to your monitor's screen coordinates
'''
def realCoord(x, y):
    rx = int(x)
    ry = int(y)
    rx, ry = sanitizeMargins(int(x), int(y))

    rx += gameScreen['left']
    ry += gameScreen['top']

    return rx, ry

def gameCoord(x, y):
    rx = int(x) - gameScreen['left']
    ry = int(y) - gameScreen['top']
    return sanitizeMargins(rx, ry)

'''
Moves mouse to (x,y) in screen coordinates
'''
def moveMouse(x, y):
    # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, 
    #     int(x/screenWidth*65535.0), int(y/screenHeight*65535.0))
    print("move mouse")

def distPoints(x1, y1, x2, y2):
    return  math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )

with mss.mss() as sct:
    while True:
        last_time = time.time()
        screen = np.array(sct.grab(gameScreen))
        # print(screen)
        screen = np.flip(screen[:, :, :3], 2) 
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        img = screen.copy()
        print(img.shape)


        mask = None
        tmpMask = None


        #cv2.imshow('result', tmpMask)
        cv2.imshow('debug', debug)
        cv2.imshow('img', img)
        
        if writeVideo:
            outResult.write(img)
            outScreen.write(screen)            
            outMask.write(debug)


        cv2.waitKey(1)
        # Press 'q' to quit
        # if keyboard.is_pressed('q'):
        #     cv2.destroyAllWindows()
        #     quit()
quit()