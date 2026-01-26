import numpy as np
import cv2
import os

# Load class names from absolute path
_utils_dir = os.path.dirname(os.path.abspath(__file__))
_code_dir = os.path.dirname(_utils_dir)
_classes_file = os.path.join(_code_dir, 'classes.txt')

if not os.path.exists(_classes_file):
    raise FileNotFoundError(f"classes.txt not found at {_classes_file}")

class_names = list(map(lambda x: x.strip(), open(_classes_file, 'r').readlines()))
#print(class_names)

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression with optimized vectorized operations.
    
    Args:
        boxes: Array of bounding boxes (N, 4)
        scores: Array of confidence scores (N,)
        iou_threshold: IoU threshold for suppression
        
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Sort by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    
    while len(sorted_indices) > 0:
        # Pick the box with highest score
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        
        if len(sorted_indices) == 1:
            break
        
        # Compute IoU of the picked box with the rest (vectorized)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        
        # Keep only boxes with IoU below threshold
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    
    return keep_boxes


def compute_iou(box, boxes):
    """Compute Intersection over Union (IoU) - fully vectorized.
    
    Args:
        box: Single bounding box (4,)
        boxes: Array of bounding boxes (N, 4)
        
    Returns:
        Array of IoU values (N,)
    """
    # Vectorized computation - all at once
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Avoid division by zero
    return intersection_area / np.maximum(union_area, 1e-8)


def xywh2xyxy(x):
    """Convert bounding box format (x, y, w, h) to (x1, y1, x2, y2) - optimized.
    
    Args:
        x: Bounding boxes in (x, y, w, h) format
        
    Returns:
        Bounding boxes in (x1, y1, x2, y2) format
    """
    y = np.empty_like(x)  # Pre-allocate instead of copy
    y[..., 0] = x[..., 0] - x[..., 2] * 0.5  # x1 = x - w/2 (multiplication faster than division)
    y[..., 1] = x[..., 1] - x[..., 3] * 0.5  # y1 = y - h/2
    y[..., 2] = x[..., 0] + x[..., 2] * 0.5  # x2 = x + w/2
    y[..., 3] = x[..., 1] + x[..., 3] * 0.5  # y2 = y + h/2
    return y

def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    """Draw bounding boxes and labels on image with optimized rendering.
    
    Args:
        image: Input image (H, W, C)
        boxes: List of bounding boxes
        scores: List of confidence scores
        class_ids: List of class IDs
        mask_alpha: Alpha transparency for overlay
        
    Returns:
        Image with drawn detections
    """
    if len(boxes) == 0:
        return image
    
    mask_img = image.copy()
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.001
    text_thickness = int(min([img_height, img_width]) * 0.002)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        class_id = int(class_id)
        
        # Bounds check
        if class_id < 0 or class_id >= len(colors):
            continue
        
        color = tuple(map(int, colors[class_id]))
        x1, y1, x2, y2 = box.astype(int)
        
        # Bounds clipping
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)

        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        # Draw label background
        cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
        
        # Draw label text
        cv2.putText(det_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

def draw_comparison(img1, img2, name1, name2, fontsize=2.6, text_thickness=3):
    (tw, th), _ = cv2.getTextSize(text=name1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img1, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (0, 115, 255), -1)
    cv2.putText(img1, name1,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)


    (tw, th), _ = cv2.getTextSize(text=name2, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img2, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (94, 23, 235), -1)

    cv2.putText(img2, name2,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)

    combined_img = cv2.hconcat([img1, img2])
    if combined_img.shape[1] > 3840:
        combined_img = cv2.resize(combined_img, (3840, 2160))

    return combined_img
