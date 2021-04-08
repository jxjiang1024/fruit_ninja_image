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
import inference_utils
from inference_utils import *
import tensorflow as tf
import os
import keyboard
import mouse
import cv2
import numpy as np
import os
import pyautogui
import sys
import threading
from time import sleep
import math
import keyboard
import cv2
import mediapipe as mp
import win32gui, win32api, win32con, ctypes, time
from hand_control import *
output_directory = os.path.abspath('inference_graph_2')
labelmap_path = os.path.abspath('labelmap.pbtxt')
print("*"*50)
print(labelmap_path)
print("*"*50)

model = tf.saved_model.load(output_directory)
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

screenHeight = 1080
screenWidth = 1920

# def moveMouse(x, y):
#     win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE, 
#         int(x/screenWidth*65535.0), int(y/screenHeight*65535.0))
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.5) as hands:
    with mss.mss() as sct:
        # Use the 1st monitor
        monitor = sct.monitors[1]

        # Capture a bbox using percent values
        # 1920 *1080 coordinates

        game_width = 600
        game_height = 500

        left = monitor["left"] + monitor["width"] * 9 // 100  # 5% from the left
        top = monitor["top"] + monitor["height"] * 23 // 100  # 5% from the top
        right = left + game_width  # 400px width
        lower = top + game_height  # 400px height
        bbox = (left, top, right, lower)
        # count = 0
        

        # left = monitor["left"] # + monitor["width"] * 5 // 100  # 5% from the left
        # top = monitor["top"] + monitor["height"] * 15 // 100  # 5% from the top
        # right = left + game_width  # 400px width
        # lower = top + game_height  # 400px height
        
        bbox = (left, top, right, lower)
        count = 0
        left_bound = int(0.235*game_width)
        right_bound = int(0.75*game_width)


        test = False
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            last_time = time.time()
            screen = np.array(sct.grab(bbox))
            screen = np.flip(screen[:, :, :3], 2) 
            # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            img = screen.copy()
            if test:
                cv2.imwrite(f'./save/raw_image/image_{count}.jpg',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            output_dict = inference_utils.run_inference_for_single_image(model, img)
            prev_x = 0
            iteration = 0
            for (i,tensor) in enumerate(output_dict['detection_boxes']):
                if output_dict['detection_scores'][i] > 0.4 and ((tensor[1] * img.shape[1] - prev_x) > prev_x*0.07 or (tensor[1] * img.shape[1] - prev_x) < prev_x*0.07):
                    xmin = int(tensor[1] * img.shape[1])
                    ymin = int(tensor[0] * img.shape[0])
                    xmax = int(tensor[3] * img.shape[1])
                    ymax = int(tensor[2] * img.shape[0])
                    center_x, center_y = (xmin + xmax)//2, (ymin + ymax)//2
                    # print(center_x, center_y)
                    if (center_x>left_bound) and (center_x<right_bound):
                        pyautogui.click(x=center_x+monitor["width"] * 9 // 100,y=center_y+(monitor["height"] * 23 // 100),interval=0)
                        # img = cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), 3)
                    prev_x = center_x
                    iteration+=1
                if iteration==3:
                    break
            detect(cap, hands)




            image_np_with_detections = vis_util.visualize_boxes_and_labels_on_image_array(img, output_dict['detection_boxes'],
                                  output_dict['detection_classes'],output_dict['detection_scores'], category_index, instance_masks=output_dict.get('detection_masks_reframed', None), use_normalized_coordinates=True,
                                  line_thickness=8)
            if test:
                cv2.imwrite(f'./save/bounding_boxes/image_detected_{count}.jpg', image_np_with_detections)
                time.sleep(4)
                if count == 10:
                    break
            # img = cv2.line(img,(line_1,0), (line_1,game_height), (0, 0, 255), 1)
            # img = cv2.line(img,(line_2,0), (line_2,game_height), (0, 0, 255), 1)

            # cv2.imshow('img', img)
            cv2.imshow("img", image_np_with_detections)
            # if cv2.waitKey(5) & 0xFF == 27:
            #     cv2.destroyAllWindows() 
            
            cv2.waitKey(1)
            count += 1 
            
            if keyboard.is_pressed("q"):
                print("break")
                break
            if keyboard.is_pressed("w"):
                print("time paused")
                # mouse.wait(10)
                while True:
                    time.sleep(1)
                    if keyboard.is_pressed("w"):
                        break
                    
                print("10 seconds till resume")
                time.sleep(10)

                
                
    
quit()