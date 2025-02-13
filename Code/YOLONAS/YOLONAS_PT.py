import os
import sys
sys.path.append('../../yolov7-custom')
sys.path.append("D:\AI-ML\AI-ML-Playground\yolov7-custom")

import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class YOLONAS_PT:

    def __init__(self, weights="yolov7-tiny.pt", yaml="data.yaml", img_size=480,conf_thres=0.65, iou_thres=0.35, device="cpu", filter_classes=None):
        self.weights=weights # Path to weights file default weights are for nano model
        self.yaml=yaml
        self.imgsz=img_size # default image size
        self.conf_thres = conf_thres # confidence threshold for inference.
        self.iou_thres = iou_thres # NMS IoU threshold for inference.
        self.device=device  # device to run our model i.e. 0 or 0,1,2,3 or cpu
        self.classes_to_filter = filter_classes # list of classes to filter or None
        #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]
        self.initialize_model()

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self):
        with torch.no_grad():
            set_logging()
            self.device = select_device(self.device)
            self.half = self.device.type != 'cpu'
            self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
            self.stride = int(self.model.stride.max())  # model stride
            imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
            if self.half:
                self.model.half()
            self.names = model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))
            return

    def detect_objects(self, image):

        with torch.no_grad():
            #img0 = cv2.imread(image)
            img = self.letterbox(image, self.imgsz, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment= False)[0]

            # Apply NMS
            classes = None
            if self.classes_to_filter:
                classes = []
                for class_name in self.classes_to_filter:
                    classes.append(self.classes_to_filter.index(class_name))

            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes= classes, agnostic= False)
            t2 = time_synchronized()
            #print(f"Inference time: {(t2-t1)*1000} ms")
            for i, det in enumerate(pred):
                s = ''
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(image.shape)[[1, 0, 1, 0]]
                if len(det):
                  det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                  for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                  for *xyxy, conf, cls in reversed(det):

                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, image, label=label, color=self.colors[int(cls)], line_thickness=1)
        return image

    def letterbox(self, img, new_shape=(480, 320), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/person.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='models/yolov7_640x640.onnx',
                        choices=["models/yolov7_640x640.onnx", "models/yolov7-tiny_640x640.onnx",
                                 "models/yolov7_736x1280.onnx", "models/yolov7-tiny_384x640.onnx",
                                 "models/yolov7_480x640.onnx", "models/yolov7_384x640.onnx",
                                 "models/yolov7-tiny_256x480.onnx", "models/yolov7-tiny_256x320.onnx",
                                 "models/yolov7_256x320.onnx", "models/yolov7-tiny_256x640.onnx",
                                 "models/yolov7_256x640.onnx", "models/yolov7-tiny_480x640.onnx",
                                 "models/yolov7-tiny_736x1280.onnx", "models/yolov7_256x480.onnx"],
                        help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.3, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    args = parser.parse_args()

    # Initialize YOLOv7 object detector
    yolonas_detector = YOLONAS(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    srcimg = cv2.imread(args.imgpath)

    # Detect Objects
    boxes, scores, class_ids = yolov7_detector.detect(srcimg)

    # Draw detections
    dstimg = yolonas_detector.draw_detections(srcimg, boxes, scores, class_ids)
    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
