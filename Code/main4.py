import cv2 as cv2
import numpy as np
import os
import time
import subprocess
from windowcapture import WindowCapture

def filter_Detections(results, thresh=0.5):
    """Filter detections by confidence threshold using vectorized operations.
    
    Args:
        results: Detection results array
        thresh: Confidence threshold (default: 0.5)
        
    Returns:
        Filtered detections
    """
    # Vectorized processing - no Python loops
    if len(results[0]) == 5:
        # Single class: results already in correct format
        mask = results[:, 4] > thresh
        return results[mask]
    else:
        # Multiple classes: extract class_id and max confidence
        class_ids = np.argmax(results[:, 4:], axis=1)
        confidences = np.max(results[:, 4:], axis=1)
        
        # Filter by confidence
        mask = confidences > thresh
        
        # Combine filtered results
        if np.any(mask):
            filtered = results[mask, :4]
            class_ids_filtered = class_ids[mask]
            conf_filtered = confidences[mask]
            return np.column_stack((filtered, class_ids_filtered, conf_filtered))
        else:
            return np.array([])

def NMS(boxes, conf_scores, iou_thresh=0.55):
    """Apply Non-Maximum Suppression using vectorized operations.
    
    Args:
        boxes: Bounding boxes array of shape (N, 4) [[x1,y1, x2,y2], ...]
        conf_scores: Confidence scores array of shape (N,)
        iou_thresh: IoU threshold for suppression (default: 0.55)
        
    Returns:
        Tuple of (keep_boxes, keep_confidences)
    """
    if len(boxes) == 0:
        return [], []
    
    # Extract coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate areas once
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by confidence (descending)
    order = np.argsort(conf_scores)[::-1]
    
    keep = []
    keep_confidences = []
    
    while len(order) > 0:
        idx = order[0]
        keep.append(boxes[idx])
        keep_confidences.append(conf_scores[idx])
        
        if len(order) == 1:
            break
        
        order = order[1:]
        
        # Vectorized IoU calculation
        xx1 = np.maximum(x1[idx], x1[order])
        yy1 = np.maximum(y1[idx], y1[order])
        xx2 = np.minimum(x2[idx], x2[order])
        yy2 = np.minimum(y2[idx], y2[order])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        union = areas[idx] + areas[order] - intersection
        iou = intersection / union
        
        # Keep only boxes with IoU below threshold
        order = order[iou < iou_thresh]
    
    return keep, keep_confidences



def rescale_back(results, img_w, img_h):
    """Vectorized rescaling of detection results back to original image dimensions."""
    if results.size == 0:
        return [], []
    
    # Vectorized scaling (all detections at once)
    cx = (results[:, 0] / 480.0) * img_w
    cy = (results[:, 1] / 320.0) * img_h
    w = (results[:, 2] / 480.0) * img_w
    h = (results[:, 3] / 320.0) * img_h
    
    # Convert center format to corner format
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    # Stack and extract confidences in one go
    class_id = results[:, 4]
    confidence = results[:, -1]
    
    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    keep, keep_confidences = NMS(boxes, confidence)
    
    return keep, keep_confidences

def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    """Draw bounding boxes and labels on image.
    
    Args:
        image: Input image (H, W, C)
        boxes: List of bounding boxes
        scores: List of confidence scores
        class_ids: List of class IDs
        mask_alpha: Alpha transparency for overlay
        
    Returns:
        Image with drawn detections
    """
    mask_img = image.copy()
    det_img = image.copy()
    colors = _get_colors()  # Lazy-load colors

    img_height, img_width = image.shape[:2]
    size = min(img_height, img_width) * 0.001
    text_thickness = max(1, int(min(img_height, img_width) * 0.002))

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        class_id_int = int(class_id)
        
        # Bounds check
        if class_id_int < 0 or class_id_int >= len(class_names):
            continue
        
        color = tuple(map(int, colors[class_id_int]))
        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = class_names[class_id_int]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

# Get the directory where this script is located
_code_dir = os.path.dirname(os.path.abspath(__file__))
_classes_path = os.path.join(_code_dir, 'classes.txt')

# Load class names with error handling
if not os.path.exists(_classes_path):
    raise FileNotFoundError(f"Classes file not found: {_classes_path}")

class_names = list(map(lambda x: x.strip(), open(_classes_path, 'r').readlines()))

if not class_names:
    raise ValueError("No class names found in classes.txt")

# Lazy-load colors - only generated when needed
_colors = None

def _get_colors():
    """Lazy-load color palette for classes."""
    global _colors
    if _colors is None:
        rng = np.random.default_rng(3)
        _colors = rng.uniform(0, 255, size=(len(class_names), 3)).astype(np.uint8)
    return _colors

print("YoloV11 ONNX with framegrabber, OpenCV(cv2) inference")

# Load trained ONNX model with error handling
model_path = os.path.join(_code_dir, 'models/yolov11n.onnx')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

try:
    # Load with explicit CPU backend to avoid CUDA validation errors
    net = cv2.dnn.readNetFromONNX(model_path)
    print(f"Model loaded successfully from {model_path}")
    
    # Set CPU backend explicitly (safest, most compatible)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("✓ Using CPU backend for inference")
    
    # Try GPU acceleration if available (CUDA). OpenCV must be built with CUDA support
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("✓ GPU acceleration enabled (CUDA)")
    except cv2.error:
        # CUDA not available, already set to CPU above
        print("⚠ CUDA unavailable, CPU backend active")
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX model: {str(e)}")

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#WindowCapture.list_window_names()
# initialize the WindowCapture class
try:
    wincap = WindowCapture('Mario Kart - Super Circuit (Europe) [Gameboy Advance] - BizHawk')
except:
    wincap=False

if (wincap==False):
    p=subprocess.Popen([r'..\\BizHawk-2.9.1-win-x64\\EmuHawk.exe',
                        r'..\\BizHawk-2.9.1-win-x64\\ROMS\\Mario Kart - Super Circuit.gba',
                        '--load-slot=1'
                        ], )

while (wincap==False):
    try:
        wincap = WindowCapture('Mario Kart - Super Circuit (Europe) [Gameboy Advance] - BizHawk')
    except:
        time.sleep(1.0)
        continue

mask = cv2.imread('mask4.jpg')
if mask is None:
    raise FileNotFoundError("mask4.jpg not found. Please check if the file exists.")

# Cache for resized mask to avoid resizing every frame
_cached_mask = mask
_cached_mask_shape = mask.shape

loop_time = time.time()
while(True):

    # get an updated image of the game
    screenshot_raw = wincap.get_screenshot()

    screenshot = np.array(screenshot_raw)
    if (mask.shape != screenshot.shape):
        print(f"Screen size changed: {screenshot.shape}")
        # Use cached mask if already resized for this size
        if _cached_mask_shape != (screenshot.shape[0], screenshot.shape[1]):
            _cached_mask = cv2.resize(mask, (screenshot.shape[1], screenshot.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            _cached_mask_shape = (screenshot.shape[0], screenshot.shape[1])
    else:
        _cached_mask = mask
    
    screenshot_masked = cv2.bitwise_and(screenshot, _cached_mask, mask=None)

    img_height, img_width = screenshot_masked.shape[:2]

    # Optimized preprocessing: chain operations with minimal copies
    img = cv2.resize(screenshot_masked, (480, 320), interpolation=cv2.INTER_LINEAR)
    
    # Normalize and transpose in one operation: (H,W,C) -> (1,C,H,W)
    img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, :]
    
    # Feed the model with processed image
    net.setInput(img)

    # Run the inference with error handling
    try:
        out = net.forward()
    except cv2.error as e:
        # If forward pass fails with backend mismatch, reset to CPU and retry
        if "preferableBackend" in str(e):
            print("Backend validation failed, resetting to CPU...")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            out = net.forward()
        else:
            raise

    results = out[0]
    results = results.transpose()
    results = filter_Detections(results)

    if (results.shape != (0,)):
        rescaled_results, confidences = rescale_back(results, img_width, img_height)

        # Batch drawing operations
        colors = _get_colors()  # Lazy-load once
        for res, conf in zip(rescaled_results, confidences):
            x1, y1, x2, y2, cls_id = res
            cls_id = int(cls_id)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf_str = f"{conf:.2f}"
            
            # Bounds check
            if 0 <= cls_id < len(class_names):
                color = tuple(map(int, colors[cls_id]))
                cv2.rectangle(screenshot, (x1, y1), (x2, y2), color, 1)
                cv2.putText(screenshot, class_names[cls_id] + ' ' + conf_str, 
                           (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    combined_img=screenshot

    fps='{:.0f} fps'.format(1 / (time.time() - loop_time))
    cv2.putText(combined_img, fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    combined_img= cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('output',combined_img)
    
    # debug the loop rate
    #print('FPS {}'.format(1 / (time.time() - loop_time)))
    loop_time = time.time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    key = cv2.waitKey(1)
    if key == ord('f'):
        pass  # Debug hook
    elif key == ord('q'):
        cv2.destroyAllWindows()
        if 'p' in locals():
            p.terminate()
        break

print('Done.')
