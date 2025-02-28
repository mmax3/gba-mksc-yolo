# Author: Sihab Sahariar
# Date: 2024-10-21
# License: MIT License
# Email: sihabsahariarcse@gmail.com
# modified by me

import argparse
import os
import sys
import os.path as osp
import cv2
import numpy as np
import onnxruntime as ort
from math import exp

#class_names = ['wario', 'toad', 'yoshi', 'tree', 'bowser', 'luigi', 'peach', 'donkey kong', 'power up', 'coin' ]
class_names = list(map(lambda x: x.strip(), open('./classes.txt', 'r').readlines()))

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


class YOLODetector:
    def __init__(self, model_path='./yolov11n-dynamic.onnx', conf_thresh=0.35, iou_thresh=0.45):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.ort_session = ort.InferenceSession(self.model_path)

        # Get input information
        self.input_details = self.ort_session.get_inputs()[0]  # First input tensor
        self.input_name = self.input_details.name
        self.input_shape = self.input_details.shape  # Extract shape

        # Check which dimensions are dynamic (None or strings)
        self.has_dynamic_shape = any(isinstance(dim, str) or dim is None for dim in self.input_details.shape)

        if self.has_dynamic_shape:
            print("Model has dynamic input, can accept various image sizes")
            self.model_input_imgH = 0
            self.model_input_imgW = 0

        else:
            print(f"Model accepts input shape: {self.input_shape}")
            self.model_input_imgH = self.input_shape[2]
            self.model_input_imgW = self.input_shape[3]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def preprocess_image(img_src, resize_w, resize_h):
        # if model is not exported with parameter dynamic=True, we need to change the size of the image,
        # accordingly how the model was exported/trained
        image = cv2.resize(img_src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        #yolo needs swapped B and R channels,
		#but we do not needed, because it is already swapped when it comes here
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        # image come in shape (320,480,3) and we need to change it to (1,3,320,480), that is the size of the models input layer
        # (to see it, open model in Netron)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def iou(self, xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
        xmin = max(xmin1, xmin2)
        ymin = max(ymin1, ymin2)
        xmax = min(xmax1, xmax2)
        ymax = min(ymax1, ymax2)

        innerWidth = max(0, xmax - xmin)
        innerHeight = max(0, ymax - ymin)

        innerArea = innerWidth * innerHeight
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        total = area1 + area2 - innerArea

        return innerArea / total

    def nms(self, detectResult):
        predBoxs = []
        
        # Sort detections by confidence score (descending order)
        sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

        for i in range(len(sort_detectboxs)):
            if sort_detectboxs[i].classId == -1:
                continue  # Skip already suppressed boxes
            
            predBoxs.append(sort_detectboxs[i])

            # Check for duplicate class detections
            for j in range(i + 1, len(sort_detectboxs)):
                if sort_detectboxs[j].classId == -1:
                    continue  # Skip already suppressed boxes

                if sort_detectboxs[i].classId == sort_detectboxs[j].classId:
                    iou = self.iou(
                        sort_detectboxs[i].xmin, sort_detectboxs[i].ymin,
                        sort_detectboxs[i].xmax, sort_detectboxs[i].ymax,
                        sort_detectboxs[j].xmin, sort_detectboxs[j].ymin,
                        sort_detectboxs[j].xmax, sort_detectboxs[j].ymax
                    )
                    
                    if iou > self.iou_thresh:
                        sort_detectboxs[j].classId = -1  # Suppress overlapping box

        return predBoxs

    def postprocess(self, out, img_h, img_w):
        # we get results in shape 1x3150x14
        outputs = out[0]
        # swap rows with columns
        outputs = outputs.transpose()
        # so now we have 3150 rows, each row is one detection
        # and each row has 14 parameters
        detectResult = []

        if not self.has_dynamic_shape:
            scale_h = img_h / self.model_input_imgH
            scale_w = img_w / self.model_input_imgW
        else:
            scale_h = 1
            scale_w = 1

        for output in outputs:

            bboxes = output[:4].squeeze() # first 4 parameters
            confidences = output[4:].squeeze()  # rest 10 parameters
            
            conf = confidences.max()
            if conf > self.conf_thresh:
                classes_score = confidences
                class_id = confidences.argmax()

                bboxes = self.xywh2xyxy(bboxes)

                x1,y1,x2,y2 = bboxes
                
                xmin = max(0, x1 * scale_w)
                ymin = max(0, y1 * scale_h)
                xmax = min(img_w, x2 * scale_w)
                ymax = min(img_h, y2 * scale_h)

                box = DetectBox(class_id, conf, xmin, ymin, xmax, ymax)
                #box = DetectBox(class_id, conf, x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h)
                detectResult.append(box)

        predBox = self.nms(detectResult)
        return predBox

    def detect(self, img_path):
        if isinstance(img_path, str):
            orig = cv2.imread(img_path)
        else:
            orig = img_path
        input_imgH,input_imgW = orig.shape[:2]
        # adjust image to fit the models input (1,3,320,480)
        if self.has_dynamic_shape:
            #print(input_imgH,input_imgW)
            image = self.preprocess_image(orig, input_imgW, input_imgH)
        else:
            image = self.preprocess_image(orig, self.model_input_imgW, self.model_input_imgH)
        # if not exported as dynamic and not trained differently, it has input dimensions [1,3,320,480],
        # parameter at onnx export was imgsz=[320, 480]
        pred_results = self.ort_session.run(None, {self.input_name: image})
        predbox = self.postprocess(pred_results, input_imgH, input_imgW)

        boxes = []
        scores = []
        class_ids = []

        for box in predbox:
            boxes.append([int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)])
            scores.append(box.score)
            class_ids.append(box.classId)

        return boxes, scores, class_ids


    def draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0.3):
        mask_img = image.copy()
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        #size = min([img_height, img_width]) * 0.0006
        size = min([img_height, img_width]) * 0.001
        #text_thickness = int(min([img_height, img_width]) * 0.001)
        text_thickness = int(min([img_height, img_width]) * 0.002)

        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = colors[class_id]

            x1, y1, x2, y2 = box#.astype(int)

            # Draw rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Draw fill rectangle in mask image
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            label = class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(det_img, (x1, y1),
                          (x1 + tw, y1 - th), color, -1)
            cv2.rectangle(mask_img, (x1, y1),
                          (x1 + tw, y1 - th), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)
    
    def xywh2xyxy(self,x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
