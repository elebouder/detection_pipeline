from object_detection.utils import visualization_utils as vis_utils
from object_detection.utils import label_map_util
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import tensorflow as tf

cwd = os.getcwd()

PATH_TO_IMAGE = cwd + '/vis/[9.75037548].png'
PATH_TO_CKPT = cwd + '/frozen_model/frozen_inference_graph.pb'
PATH_TO_LABELS = cwd + '/lsat_label_map.pbtxt'
NUM_CLASSES = 1


# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)
(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})

print PATH_TO_IMAGE
print ([category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5])

vis_utils.visualize_boxes_and_labels_on_image_array(image,
                                                        np.squeeze(boxes),
                                                        np.squeeze(classes).astype(np.int32),
                                                        np.squeeze(scores),
                                                        category_index,
                                                        use_normalized_coordinates=True,
                                                        line_thickness=8,
                                                        min_score_thresh=0.80)


#cv2.imshow('detection', image)
cv2.imwrite(cwd + '/test.png', image)
