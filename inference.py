from inference_utils import *

# from utils import label_map_util
# from utils import visualization_utils as vis_util
import tensorflow as tf
import cv2

output_directory = 'inference_graph_2'
labelmap_path = 'C:/Users/Kai/Desktop/UI/labelmap.pbtxt'

category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
tf.keras.backend.clear_session()
model = tf.saved_model.load(f'C:/Users/Kai/Desktop/UI/{output_directory}')

import pandas as pd
# test = pd.read_csv('dataset/test_labels.csv')
# images = list(test['filename'][0:20])

path = "C:/Users/Kai/Desktop/UI/thao/"


images = ['frame-thao (1).jpg', 'frame-thao (2).jpg', 'frame-thao (3).jpg']

for image_name in images:
  new_path = 'C:/Users/Kai/Desktop/UI/thao/prediction_{}'.format(image_name)
  image_np = load_image_into_numpy_array(path + image_name)
  output_dict = run_inference_for_single_image(model, image_np)
  image_np_with_detections = vis_util.visualize_boxes_and_labels_on_image_array(
                              image_np,
                              output_dict['detection_boxes'],
                              output_dict['detection_classes'],
                              output_dict['detection_scores'],
                              category_index,
                              instance_masks=output_dict.get('detection_masks_reframed', None),
                              use_normalized_coordinates=True,
                              line_thickness=8)
  # display(Image.fromarray(image_np))
  cv2.imwrite(new_path,cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR))