import cv2 as cv2
import numpy as np
import os
import time
import subprocess
#import pyautogui
#from PIL import Image
from windowcapture import WindowCapture

print("YoloV11 ONNX with framegrabber, openCV(cv2) inference")

def filter_Detections(results, thresh = 0.5):
    # if model is trained on 1 class only
    if len(results[0]) == 5:
        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in results if detection[4] > thresh]
        considerable_detections = np.array(considerable_detections)
        return considerable_detections

    # if model is trained on multiple classes
    else:
        A = []
        for detection in results:

            class_id = detection[4:].argmax()
            confidence_score = detection[4:].max()

            new_detection = np.append(detection[:4],[class_id,confidence_score])

            A.append(new_detection)

        A = np.array(A)

        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in A if detection[-1] > thresh]
        considerable_detections = np.array(considerable_detections)

        return considerable_detections

def NMS(boxes, conf_scores, iou_thresh = 0.55):

    #  boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        xx1 = np.take(x1, indices= order)
        yy1 = np.take(y1, indices= order)
        xx2 = np.take(x2, indices= order)
        yy2 = np.take(y2, indices= order)

        keep.append(A)
        keep_confidences.append(conf)

        # iou = inter/union

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2-xx1, 0)
        h = np.maximum(yy2-yy1, 0)

        intersection = w*h

        # union = areaA + other_areas - intesection
        other_areas = np.take(areas, indices= order)
        union = areas[idx] + other_areas - intersection

        iou = intersection/union

        boleans = iou < iou_thresh

        order = order[boleans]

        # order = [2,0,1]  boleans = [True, False, True]
        # order = [2,1]

    return keep, keep_confidences



def rescale_back(results,img_w,img_h):
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
    cx = cx/640.0 * img_w
    cy = cy/640.0 * img_h
    w = w/640.0 * img_w
    h = h/640.0 * img_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    keep, keep_confidences = NMS(boxes,confidence)
    #print(np.array(keep).shape)
    return keep, keep_confidences

def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
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

        x1, y1, x2, y2 = box.astype(int)

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

#class_names = ['wario', 'toad', 'yoshi', 'tree', 'bowser', 'luigi', 'peach', 'donkey kong', 'power up', 'coin' ]
class_names = list(map(lambda x: x.strip(), open('./classes.txt', 'r').readlines()))
#print(class_names)

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

print("YoloV11 ONNX with framegrabber, OpenCV(cv2) inference")

# read the trained onnx model
net = cv2.dnn.readNetFromONNX('models/yolov11n.onnx')  # readNet() also works

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
    p=subprocess.Popen([r'..\\BizHawk-win-x64\\EmuHawk.exe',
                        r'..\\BizHawk-win-x64\\ROMS\\Mario Kart - Super Circuit.gba',
                        '--load-slot=1'
                        ], )

while (wincap==False):
    try:
        wincap = WindowCapture('Mario Kart - Super Circuit (Europe) [Gameboy Advance] - BizHawk')
    except:
        time.sleep(1.0)
        continue

mask = cv2.imread('mask4.jpg')

loop_time = time.time()
while(True):

    # get an updated image of the game
    screenshot_raw = wincap.get_screenshot()

    screenshot = np.array(screenshot_raw)
    if (mask.shape!=screenshot.shape):
        print(f"Wrong screen size: {screenshot.shape}")
        mask = cv2.resize(mask, (screenshot.shape[1], screenshot.shape[0]), interpolation = cv2.INTER_NEAREST)
    screenshot_masked=cv2.bitwise_and(screenshot,mask,mask=None)

    img=screenshot_masked

    # YOLOv11 need RGB image
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_height,img_width = img.shape[:2]

    # resize image to get the desired size (640,640) for inference
    img = cv2.resize(img,(480,320))

    # change the order of image dimension from (640,640,3) to (3,640,640)
    img = img.transpose(2,0,1)

    # add an extra dimension at index 0
    img = img.reshape(1,3,320,480)

    # scale to 0-1
    img = img/255.0
    
    #img = cv2.dnn.blobFromImage(screenshot, scalefactor=1 / 255, size=(480, 320), swapRB=True)
    
    # feed the model with processed image
    net.setInput(img)

    # run the inference
    out = net.forward()

    results = out[0]
    results = results.transpose()
    results = filter_Detections(results)

    if (results.shape != (0,)):
        rescaled_results, confidences = rescale_back(results, img_width, img_height)

        for res, conf in zip(rescaled_results, confidences):

            x1,y1,x2,y2, cls_id = res
            cls_id = int(cls_id)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = "{:.2f}".format(conf)
            # draw the bounding boxes
            color = colors[cls_id]
            cv2.rectangle(screenshot,(int(x1),int(y1)),(int(x2),int(y2)),color,1)
            cv2.putText(screenshot,class_names[cls_id]+' '+conf,(x1,y1-17),
                        cv2.FONT_HERSHEY_SIMPLEX,1,color,1)

    combined_img=screenshot

    fps='{:.0f} fps'.format(1 / (time.time() - loop_time))
    cv2.putText(combined_img, fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    combined_img= cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('output',screenshot)
    
    # debug the loop rate
    #print('FPS {}'.format(1 / (time.time() - loop_time)))
    loop_time = time.time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    
    if cv2.waitKey(1) == ord('f'):
        pass
    elif cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        p.terminate()
        break

print('Done.')
