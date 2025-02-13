import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from YOLONAS import YOLONAS
from YOLOv7 import YOLOv7

# Initialize YOLOv7 object detector
model_path = "yolov7-tiny.onnx"
yolo_detector = YOLOv7(model_path, conf_thres=0.35, iou_thres=0.65)

# Read image
#img = cv2.imread('debug.bmp')
#img = plt.imread('debug.bmp')
img= np.array(Image.open('Mario Kart - Super Circuit-1.png'))

# converted from BGR to RGB
#color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

# Detect Objects
boxes, scores, class_ids = yolo_detector(img)
#print(len(boxes),boxes[0],scores[0],class_ids[0])

# Draw detections
combined_img = yolo_detector.draw_detections(img)
combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", combined_img)
#cv2.imwrite("detected_objects.jpg", combined_img)
cv2.waitKey(0)
