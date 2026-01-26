# Author: Sihab Sahariar
# Date: 2024-10-21
# License: MIT License
# Email: sihabsahariarcse@gmail.com
# modified by me

import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
from math import exp

# Get the directory where this module is located
_module_dir = os.path.dirname(os.path.abspath(__file__))
# Get the Code directory (parent of yolo11)
_code_dir = os.path.dirname(_module_dir)
# Path to classes.txt in the Code directory
_classes_path = os.path.join(_code_dir, 'classes.txt')

# Load class names once at module level
if not os.path.exists(_classes_path):
    raise FileNotFoundError(f"Classes file not found: {_classes_path}")

with open(_classes_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

if not class_names:
    raise ValueError("No class names found in classes.txt")

# Colors will be generated lazily when needed
_colors = None

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


class YOLODetector:
    """YOLO object detector using ONNX Runtime.
    
    Attributes:
        model_path (str): Path to ONNX model file
        conf_thresh (float): Confidence threshold for detections (0.0-1.0)
        iou_thresh (float): IoU threshold for NMS (0.0-1.0)
        mask_alpha (float): Alpha value for visualization mask overlay
    """
    
    def __init__(self, model_path='./yolov11n-dynamic.onnx', conf_thresh=0.35, iou_thresh=0.45, mask_alpha=0.3):
        """Initialize YOLO detector.
        
        Args:
            model_path (str): Path to ONNX model file
            conf_thresh (float): Confidence threshold (default: 0.35)
            iou_thresh (float): IoU threshold for NMS (default: 0.45)
            mask_alpha (float): Alpha for visualization (default: 0.3)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If ONNX Runtime fails to load model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.mask_alpha = mask_alpha
        
        try:
            self.ort_session = ort.InferenceSession(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}")

        # Get input information
        if not self.ort_session.get_inputs():
            raise RuntimeError("ONNX model has no input tensors")
        
        self.input_details = self.ort_session.get_inputs()[0]
        self.input_name = self.input_details.name
        self.input_shape = self.input_details.shape

        # Check which dimensions are dynamic (None or strings)
        self.has_dynamic_shape = any(isinstance(dim, str) or dim is None for dim in self.input_shape)

        if self.has_dynamic_shape:
            print("[YOLO] Model has dynamic input, accepts various image sizes")
            self.model_input_imgH = 0
            self.model_input_imgW = 0
        else:
            print(f"[YOLO] Model input shape: {self.input_shape}")
            if len(self.input_shape) != 4:
                raise ValueError(f"Expected 4D input shape, got {self.input_shape}")
            self.model_input_imgH = self.input_shape[2]
            self.model_input_imgW = self.input_shape[3]

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + exp(-x))

    @staticmethod
    def preprocess_image(img_src, resize_w, resize_h):
        """Preprocess image for YOLO model input.
        
        Resizes image, normalizes to [0, 1], and transposes to (1, 3, H, W) format.
        
        Args:
            img_src: Input image (HxWx3)
            resize_w: Target width
            resize_h: Target height
            
        Returns:
            Preprocessed image in shape (1, 3, resize_h, resize_w)
        """
        image = cv2.resize(img_src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        # Transpose from (H, W, 3) to (3, H, W) then add batch dimension (1, 3, H, W)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def iou(self, xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            Coordinates of two boxes in format (xmin, ymin, xmax, ymax)
            
        Returns:
            float: IoU value between 0 and 1
        """
        xmin = max(xmin1, xmin2)
        ymin = max(ymin1, ymin2)
        xmax = min(xmax1, xmax2)
        ymax = min(ymax1, ymax2)

        inner_width = max(0, xmax - xmin)
        inner_height = max(0, ymax - ymin)
        inner_area = inner_width * inner_height
        
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        union_area = area1 + area2 - inner_area

        return inner_area / union_area if union_area > 0 else 0

    def nms(self, detect_result):
        """Apply Non-Maximum Suppression (NMS) to filter overlapping detections.
        
        Args:
            detect_result: List of DetectBox objects
            
        Returns:
            List of DetectBox objects after NMS filtering
        """
        if not detect_result or len(detect_result) <= 1:
            return detect_result
        
        # Sort detections by confidence score (descending order)
        sort_boxes = sorted(detect_result, key=lambda x: x.score, reverse=True)
        pred_boxes = []
        
        # Mark suppressed boxes by setting classId to -1 (faster than set tracking)
        for i in range(len(sort_boxes)):
            if sort_boxes[i].classId == -1:
                continue  # Skip already suppressed boxes
            
            pred_boxes.append(sort_boxes[i])
            
            # Suppress overlapping boxes of same class
            for j in range(i + 1, len(sort_boxes)):
                if sort_boxes[j].classId == -1:
                    continue  # Skip already suppressed
                
                if sort_boxes[i].classId == sort_boxes[j].classId:
                    iou_val = self.iou(
                        sort_boxes[i].xmin, sort_boxes[i].ymin,
                        sort_boxes[i].xmax, sort_boxes[i].ymax,
                        sort_boxes[j].xmin, sort_boxes[j].ymin,
                        sort_boxes[j].xmax, sort_boxes[j].ymax
                    )
                    
                    if iou_val > self.iou_thresh:
                        sort_boxes[j].classId = -1
        
        return pred_boxes

    def postprocess(self, out, img_h, img_w):
        """Postprocess YOLO model output to extract detections.
        
        Args:
            out: Model output tensor from ONNX inference
            img_h: Original image height
            img_w: Original image width
            
        Returns:
            List of DetectBox objects after NMS
            
        Raises:
            ValueError: If output shape is invalid
        """
        if not out or len(out) == 0:
            return []
        
        outputs = out[0]  # Get first output from ONNX session
        
        # Handle different output shapes efficiently
        if outputs.ndim == 3:
            # Shape: (1, channels, detections) - squeeze batch dimension
            outputs = outputs.squeeze(0)
        elif outputs.ndim != 2 and outputs.ndim != 3:
            # Try to reshape for unexpected shapes
            try:
                outputs = outputs.reshape(-1, outputs.shape[-1])
            except Exception as e:
                raise ValueError(f"Cannot process output shape {outputs.shape}: {str(e)}")
        
        # Transpose only if needed (channels should be smaller than detections)
        if outputs.ndim == 3 or (outputs.shape[0] < outputs.shape[1]):
            outputs = outputs.transpose() if outputs.ndim == 2 else outputs.squeeze(0).transpose()
        
        detect_result = []
        
        # Calculate scale factors for coordinate conversion
        if not self.has_dynamic_shape:
            scale_h = img_h / self.model_input_imgH
            scale_w = img_w / self.model_input_imgW
        else:
            scale_h = 1.0
            scale_w = 1.0
        
        # Vectorized processing: extract bboxes and confidences
        bboxes = outputs[:, :4]
        confidences = outputs[:, 4:]
        
        # Get max confidence and class for all detections at once
        confs = np.max(confidences, axis=1)
        class_ids = np.argmax(confidences, axis=1)
        
        # Create mask for detections above threshold
        mask = confs > self.conf_thresh
        
        # Process only detections above threshold
        valid_bboxes = bboxes[mask]
        valid_confs = confs[mask]
        valid_classes = class_ids[mask]
        
        for bbox, conf, class_id in zip(valid_bboxes, valid_confs, valid_classes):
            # Convert from xywh to xyxy format
            x1, y1, x2, y2 = self.xywh2xyxy(bbox)
            
            # Scale coordinates and clip to image bounds
            xmin = max(0, x1 * scale_w)
            ymin = max(0, y1 * scale_h)
            xmax = min(img_w, x2 * scale_w)
            ymax = min(img_h, y2 * scale_h)
            
            box = DetectBox(class_id, conf, xmin, ymin, xmax, ymax)
            detect_result.append(box)
        
        # Apply NMS to filter overlapping detections
        pred_boxes = self.nms(detect_result)
        return pred_boxes

    def detect(self, img_path):
        """Run object detection on an image.
        
        Args:
            img_path: Either file path (str) or numpy array (HxWx3)
            
        Returns:
            Tuple of (boxes, scores, class_ids) where:
                - boxes: List of [x1, y1, x2, y2] coordinates
                - scores: List of confidence scores
                - class_ids: List of class IDs
                
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        try:
            if isinstance(img_path, str):
                orig = cv2.imread(img_path)
                if orig is None:
                    raise ValueError(f"Failed to load image: {img_path}")
            else:
                orig = img_path
                if not isinstance(orig, np.ndarray):
                    raise ValueError("Image must be numpy array or file path")
            
            input_imgH, input_imgW = orig.shape[:2]
            
            # Preprocess image to model input size
            if self.has_dynamic_shape:
                image = self.preprocess_image(orig, input_imgW, input_imgH)
            else:
                image = self.preprocess_image(orig, self.model_input_imgW, self.model_input_imgH)
            
            # Run inference
            pred_results = self.ort_session.run(None, {self.input_name: image})
            pred_boxes = self.postprocess(pred_results, input_imgH, input_imgW)
            
            # Extract boxes, scores, and class IDs
            boxes = []
            scores = []
            class_ids = []
            
            for box in pred_boxes:
                boxes.append([int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)])
                scores.append(float(box.score))
                class_ids.append(int(box.classId))
            
            return boxes, scores, class_ids
            
        except Exception as e:
            raise ValueError(f"Detection failed: {str(e)}")


    def _generate_colors(self):
        """Generate random colors for each class (lazy initialization)."""
        global _colors
        if _colors is None:
            rng = np.random.default_rng(3)
            _colors = rng.uniform(0, 255, size=(len(class_names), 3)).astype(np.uint8)
        return _colors
    
    def draw_detections(self, image, boxes, scores, class_ids, mask_alpha=None):
        """Draw bounding boxes and labels on image.
        
        Args:
            image: Input image (HxWx3)
            boxes: List of [x1, y1, x2, y2] coordinates
            scores: List of confidence scores
            class_ids: List of class IDs
            mask_alpha: Alpha blending value (uses instance default if None)
            
        Returns:
            Image with drawn detections
        """
        if mask_alpha is None:
            mask_alpha = self.mask_alpha
        
        if not boxes:
            return image
        
        mask_img = image.copy()
        det_img = image.copy()
        colors = self._generate_colors()
        
        img_height, img_width = image.shape[:2]
        font_scale = min([img_height, img_width]) * 0.001
        text_thickness = int(min([img_height, img_width]) * 0.002)
        
        # Draw bounding boxes and labels for each detection
        for box, score, class_id in zip(boxes, scores, class_ids):
            if class_id < 0 or class_id >= len(class_names):
                continue
            
            color = tuple(map(int, colors[class_id]))
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            
            # Create label
            label = class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            (text_w, text_h), _ = cv2.getTextSize(
                text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, thickness=text_thickness
            )
            text_h = int(text_h * 1.2)
            
            # Draw label background and text
            cv2.rectangle(det_img, (x1, y1), (x1 + text_w, y1 - text_h), color, -1)
            cv2.rectangle(mask_img, (x1, y1), (x1 + text_w, y1 - text_h), color, -1)
            cv2.putText(det_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
        
        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)
    
    def xywh2xyxy(self, x):
        """Convert bounding box from (x, y, w, h) to (x1, y1, x2, y2) format.
        
        Args:
            x: Bounding box in xywh format
            
        Returns:
            Bounding box in xyxy format
        """
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1 = center_x - w/2
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1 = center_y - h/2
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2 = center_x + w/2
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2 = center_y + h/2
        return y
