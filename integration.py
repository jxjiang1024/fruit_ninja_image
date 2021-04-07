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

output_directory = os.path.abspath('inference_graph_2')
labelmap_path = os.path.abspath('labelmap.pbtxt')

model = tf.saved_model.load(output_directory)
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)


with mss.mss() as sct:
    # Use the 1st monitor
    monitor = sct.monitors[1]

    # Capture a bbox using percent values
    left = monitor["left"] # + monitor["width"] * 5 // 100  # 5% from the left
    top = monitor["top"] + monitor["height"] * 15 // 100  # 5% from the top
    right = left + 700  # 400px width
    lower = top + 500  # 400px height
    bbox = (left, top, right, lower)
    count = 0
    test = False
    while True:
        last_time = time.time()
        screen = np.array(sct.grab(bbox))
        screen = np.flip(screen[:, :, :3], 2) 
        # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        img = screen.copy()
        if test:
            cv2.imwrite(f'./save/raw_image/image_{count}.jpg',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        output_dict = inference_utils.run_inference_for_single_image(model, img)

        image_np_with_detections = vis_util.visualize_boxes_and_labels_on_image_array(img, output_dict['detection_boxes'],
                              output_dict['detection_classes'],output_dict['detection_scores'], category_index, instance_masks=output_dict.get('detection_masks_reframed', None), use_normalized_coordinates=True,
                              line_thickness=8)
        if test:
            cv2.imwrite(f'./save/bounding_boxes/image_detected_{count}.jpg', image_np_with_detections)
            time.sleep(4)
            if count == 10:
                break

        # cv2.imshow('img', img)
        cv2.imshow("img", image_np_with_detections)
        
        cv2.waitKey(1)
        count += 1 
   
quit()