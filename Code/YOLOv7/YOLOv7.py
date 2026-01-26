import time
import cv2
import numpy as np
import onnxruntime
from .utils import draw_detections, nms
import argparse

class YOLOv7:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        """Initialize YOLOv7 detector.
        
        Args:
            path: Path to ONNX model file
            conf_thres: Confidence threshold for detections
            iou_thres: IoU threshold for NMS
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.cuda = False
        self._cached_offset = None  # Cache for rescale offset

        # Initialize model
        self.initialize_model(path, self.cuda)

    def __call__(self, image):
        """Allow calling detector as function."""
        return self.detect_objects(image)

    def initialize_model(self, path, cuda):
        """Initialize ONNX Runtime session and load model."""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(path, providers=providers)
        # Get model info
        self.get_input_details()
        self.get_output_details()
        self.has_postprocess = 'score' in self.output_names

    def detect_objects(self, image):
        """Detect objects in image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        
        if self.has_postprocess:
            self.boxes, self.scores, self.class_ids = self.parse_processed_output(outputs)
        else:
            self.boxes, self.scores, self.class_ids = self.process_output(outputs)
         
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        """Prepare input image for model inference.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Normalized image tensor ready for inference
        """
        self.img_height, self.img_width = image.shape[:2]

        # Apply letterbox resizing
        image, self.ratio, self.dwdh = self.letterbox(image, auto=False)
        
        # Pre-compute offset once instead of in rescale_boxes every frame
        self._dwdh_offset = self.dwdh * 2
        
        # Normalize and reshape in minimal operations
        image = image.astype(np.float32, copy=False) * (1.0 / 255.0)
        image = image.transpose((2, 0, 1))  # (H,W,C) -> (C,H,W), creates view not copy
        image = image[np.newaxis, ...]  # Add batch: (C,H,W) -> (1,C,H,W)
        
        return np.ascontiguousarray(image)

    def inference(self, input_tensor):
        """Run model inference.
        
        Args:
            input_tensor: Preprocessed input image
            
        Returns:
            Model output
        """
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output):
        """Process model output and extract detections.
        
        Args:
            output: Model output tensor
            
        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        predictions = output[0]
        
        # Handle different tensor dimensions
        if predictions.ndim == 0:
            return [], [], []
        elif predictions.ndim == 1:
            predictions = predictions[np.newaxis, :]
        
        # Extract scores and filter early (avoid processing non-confident detections)
        scores = np.squeeze(predictions[:, 6:], axis=1)
        valid_mask = scores > self.conf_threshold
        
        if not np.any(valid_mask):
            return [], [], []
        
        # Filter: only keep high-confidence predictions
        predictions = predictions[valid_mask]
        scores = scores[valid_mask]
        
        # Extract class_ids and boxes (vectorized)
        class_ids = predictions[:, 5].astype(int)  # Direct indexing, no squeeze
        boxes = self.extract_boxes(predictions)
        
        # Apply NMS
        indices = nms(boxes, scores, self.iou_threshold)
        
        return boxes[indices], scores[indices], class_ids[indices]

    def parse_processed_output(self, outputs):
        """Parse output from models with built-in postprocessing.
        
        Args:
            outputs: Model outputs
            
        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        scores = np.squeeze(outputs[0], axis=1)
        predictions = outputs[1]
        
        # Filter by confidence
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]

        if len(scores) == 0:
            return [], [], []

        # Extract the boxes and class ids
        batch_number = predictions[:, 0]
        class_ids = predictions[:, 1]
        boxes = predictions[:, 2:]
        
        # Swap axes (x,y are reversed in postprocess output)
        boxes = boxes[:, [1, 0, 3, 2]]
        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return boxes, scores, class_ids

    def extract_boxes(self, predictions):
        """Extract and rescale bounding boxes from predictions.
        
        Args:
            predictions: Model predictions array
            
        Returns:
            Rescaled bounding boxes
        """
        # Extract boxes from predictions (columns 1-5)
        boxes = predictions[:, 1:5]
        # Rescale boxes to original image dimensions
        return self.rescale_boxes(boxes)

    def rescale_boxes(self, boxes):
        """Rescale boxes from model input dimensions to original image dimensions.
        
        Args:
            boxes: Bounding boxes in model coordinate space
            
        Returns:
            Bounding boxes in original image coordinate space
        """
        # Use pre-computed offset from prepare_input
        boxes = boxes - self._dwdh_offset
        boxes = boxes / self.ratio  # Broadcasting faster than itemwise
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        """Draw detections on image."""
        return draw_detections(image, self.boxes, self.scores, self.class_ids, mask_alpha)

    def get_input_details(self):
        """Get model input specifications."""
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        """Get model output specifications."""
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def letterbox(self, im, new_shape=(480, 320), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        """Resize and pad image while meeting stride-multiple constraints.
        
        Args:
            im: Input image
            new_shape: Target shape (H, W)
            color: Padding color
            auto: Whether to auto-pad
            scaleup: Whether to scale up
            stride: Stride constraint
            
        Returns:
            Padded image, scale ratio, and padding offsets
        """
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]

        if auto:
            dw = np.mod(dw, stride)
            dh = np.mod(dh, stride)

        dw /= 2
        dh /= 2

        # Resize only if needed
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add padding in one operation
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return im, r, (dw, dh)

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
    yolov7_detector = YOLOv7(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    srcimg = cv2.imread(args.imgpath)

    # Detect Objects
    boxes, scores, class_ids = yolov7_detector.detect_objects(srcimg)
    
    # Draw detections
    dstimg = yolov7_detector.draw_detections(srcimg)
    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

