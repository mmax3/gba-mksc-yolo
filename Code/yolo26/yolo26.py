import cv2
import numpy as np
import onnxruntime as ort
import os

# Get the directory where this module is located
_module_dir = os.path.dirname(os.path.abspath(__file__))
# Get the Code directory (parent of yolo26)
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

# Colors will be generated lazily when first needed
_colors = None


class DetectBox:
    """Container for detection box information."""
    __slots__ = ('classId', 'score', 'xmin', 'ymin', 'xmax', 'ymax')
    
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
    """
    
    def __init__(self, model_path='./yolov26n.onnx', conf_thresh=0.35, iou_thresh=0.45):
        """Initialize YOLO detector.
        
        Args:
            model_path (str): Path to ONNX model file
            conf_thresh (float): Confidence threshold (default: 0.35)
            iou_thresh (float): Unused, kept for compatibility
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If ONNX Runtime fails to load model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self._last_scale_h = None  # Cache for last scale factor
        self._last_scale_w = None
        
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
            print("[YOLO26] Model has dynamic input, accepts various image sizes")
            self.model_input_imgH = 0
            self.model_input_imgW = 0
        else:
            print(f"[YOLO26] Model input shape: {self.input_shape}")
            if len(self.input_shape) != 4:
                raise ValueError(f"Expected 4D input shape, got {self.input_shape}")
            self.model_input_imgH = self.input_shape[2]
            self.model_input_imgW = self.input_shape[3]

    @staticmethod
    def _get_colors():
        """Get or generate color palette for classes (lazy initialization)."""
        global _colors
        if _colors is None:
            rng = np.random.default_rng(3)
            _colors = rng.uniform(0, 255, size=(len(class_names), 3)).astype(np.uint8)
        return _colors

    @staticmethod
    def preprocess_image(img_src, resize_w, resize_h):
        """Preprocess image for YOLO model input.
        
        Args:
            img_src: Input image (H, W, C) in BGR format
            resize_w: Target width
            resize_h: Target height
            
        Returns:
            Preprocessed image (1, C, H, W) in float32 [0, 1] range
        """
        # Resize image
        image = cv2.resize(img_src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) * (1.0 / 255.0)
        # Transpose from (H, W, C) to (C, H, W) and add batch dimension
        return np.expand_dims(image.transpose(2, 0, 1), axis=0)

    def postprocess(self, out, img_h, img_w):
        """Postprocess model output to extract bounding boxes.
        
        Args:
            out: Model output tensor of shape (1, N, 6) where N is max detections
            img_h: Original image height
            img_w: Original image width
            
        Returns:
            List of DetectBox objects
        """
        # Squeeze batch dimension: (1, N, 6) -> (N, 6)
        output_tensor = out[0] if isinstance(out, list) else out
        outputs = np.squeeze(output_tensor, axis=0) if output_tensor.ndim == 3 else np.squeeze(output_tensor)
        
        # Calculate scaling factors once
        if not self.has_dynamic_shape:
            scale_h = img_h / self.model_input_imgH
            scale_w = img_w / self.model_input_imgW
        else:
            scale_h = 1.0
            scale_w = 1.0

        # Extract all components at once (vectorized)
        bboxes = outputs[:, :4]  # (N, 4)
        confs = outputs[:, 4]    # (N,)
        class_ids = outputs[:, 5]  # (N,)
        
        # Filter by confidence threshold (early exit if no detections)
        valid_mask = confs > self.conf_thresh
        if not np.any(valid_mask):
            return []
        
        # Apply mask to get valid detections
        bboxes = bboxes[valid_mask]
        confs = confs[valid_mask]
        class_ids = class_ids[valid_mask]
        
        # Scale and clip coordinates (vectorized)
        xmin = np.clip(bboxes[:, 0] * scale_w, 0, img_w).astype(np.int32)
        ymin = np.clip(bboxes[:, 1] * scale_h, 0, img_h).astype(np.int32)
        xmax = np.clip(bboxes[:, 2] * scale_w, 0, img_w).astype(np.int32)
        ymax = np.clip(bboxes[:, 3] * scale_h, 0, img_h).astype(np.int32)
        
        # Create DetectBox objects (vectorized zip)
        return [
            DetectBox(int(cid), float(conf), int(xm), int(ym), int(xM), int(yM))
            for cid, conf, xm, ym, xM, yM in zip(class_ids, confs, xmin, ymin, xmax, ymax)
        ]

    def detect(self, img_path):
        """Detect objects in an image.
        
        Args:
            img_path: Path to image file or numpy array (H, W, C)
            
        Returns:
            Tuple of (boxes, scores, class_ids) where:
            - boxes: List of [xmin, ymin, xmax, ymax] coordinates
            - scores: List of confidence scores
            - class_ids: List of class IDs
            
        Raises:
            ValueError: If image cannot be loaded or processed
        """
        try:
            # Load or use provided image
            if isinstance(img_path, str):
                orig = cv2.imread(img_path)
                if orig is None:
                    raise ValueError(f"Could not load image from path: {img_path}")
            else:
                orig = img_path
            
            if orig is None or orig.size == 0:
                raise ValueError("Invalid image provided")
                
            input_imgH, input_imgW = orig.shape[:2]
            
            # Preprocess image based on model type
            if self.has_dynamic_shape:
                image = self.preprocess_image(orig, input_imgW, input_imgH)
            else:
                image = self.preprocess_image(orig, self.model_input_imgW, self.model_input_imgH)
            
            # Run inference
            pred_results = self.ort_session.run(None, {self.input_name: image})
            
            # Postprocess results
            predbox = self.postprocess(pred_results, input_imgH, input_imgW)

            # Extract results using vectorized approach (avoid nested loops)
            if predbox:
                # Use zip to avoid three separate loops
                boxes, scores, class_ids = [], [], []
                for box in predbox:
                    boxes.append([box.xmin, box.ymin, box.xmax, box.ymax])
                    scores.append(box.score)
                    class_ids.append(box.classId)
            else:
                boxes, scores, class_ids = [], [], []

            return boxes, scores, class_ids
            
        except Exception as e:
            raise ValueError(f"Detection failed: {str(e)}")


    def draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0.3):
        """Draw bounding boxes and labels on image.
        
        Args:
            image: Input image (H, W, C)
            boxes: List of [xmin, ymin, xmax, ymax] coordinates
            scores: List of confidence scores
            class_ids: List of class IDs
            mask_alpha: Alpha transparency for mask overlay
            
        Returns:
            Image with drawn detections
        """
        if not boxes:
            return image
            
        mask_img = image.copy()
        det_img = image.copy()
        colors = self._get_colors()  # Lazy-load colors once

        img_height, img_width = image.shape[:2]
        # Calculate text size and thickness based on image dimensions
        size = min(img_height, img_width) * 0.001
        text_thickness = max(1, int(min(img_height, img_width) * 0.002))
        
        # Pre-convert color indices to BGR tuples
        color_map = {int(cid): tuple(map(int, colors[int(cid)])) 
                     for cid in class_ids if int(cid) < len(class_names)}

        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            class_id_int = int(class_id)
            
            # Bounds check
            if class_id_int not in color_map:
                continue
            
            color = color_map[class_id_int]
            x1, y1, x2, y2 = box

            # Draw bounding box rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            # Prepare label text
            label = class_names[class_id_int]
            caption = f'{label} {int(score * 100)}%'
            
            # Get text size
            (tw, th), baseline = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, size, text_thickness
            )
            th = int(th * 1.2)

            # Draw label background
            label_y = max(y1, th)
            cv2.rectangle(det_img, (x1, label_y - th), (x1 + tw, label_y), color, -1)
            cv2.rectangle(mask_img, (x1, label_y - th), (x1 + tw, label_y), color, -1)
            
            # Draw label text
            cv2.putText(det_img, caption, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 
                        text_thickness, cv2.LINE_AA)
            cv2.putText(mask_img, caption, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 
                        text_thickness, cv2.LINE_AA)

        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)
