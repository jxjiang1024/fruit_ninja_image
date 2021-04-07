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
model = tf.saved_model.load(output_directory)
labelmap_path = os.path.abspath('labelmap.pbtxt')

category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
for i in range(10):
    img = cv2.imread(f'save/raw_image/image_{i}.jpg', cv2.IMREAD_COLOR)
    output_dict = inference_utils.run_inference_for_single_image(model, img)
    image_np_with_detections = vis_util.visualize_boxes_and_labels_on_image_array(img, output_dict['detection_boxes'],
                              output_dict['detection_classes'],output_dict['detection_scores'], category_index, 
                              instance_masks=output_dict.get('detection_masks_reframed', None), use_normalized_coordinates=True,
                              line_thickness=8)
    cv2.imwrite(f'save/trial/{i}.jpg', image_np_with_detections)

