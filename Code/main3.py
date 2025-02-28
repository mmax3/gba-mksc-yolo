import cv2 as cv2
import numpy as np
import os
import time
import subprocess
#import pyautogui
#from PIL import Image
from windowcapture import WindowCapture
from yolov11.yolocore import YOLODetector

print("YoloV11 ONNX with framegrabber, custom code inference")

# Initialize YOLOv11 object detector
model_path = 'models/yolov11n.onnx'
yolov11_detector =  YOLODetector(model_path= model_path , conf_thresh=0.35, iou_thresh=0.6)

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

loop_time = time.time()
while(True):

    # get an updated image of the game
    screenshot_raw = wincap.get_screenshot()
    screenshot = np.array(screenshot_raw)# už má prehodené RB kanály
    if (mask.shape!=screenshot.shape):
        print(f"Wrong screen size: {screenshot.shape}")
        # adjust mask size to fit the screenshot image
        mask = cv2.resize(mask, (screenshot.shape[1], screenshot.shape[0]), interpolation = cv2.INTER_NEAREST)
    # apply mask
    screenshot_masked=cv2.bitwise_and(screenshot,mask,mask=None)
    #cv2.imshow('input',screenshot_masked)
    
    # pre-process the image
    #apply filter
    #processed_image = vision.apply_hsv_filter(screenshot)
    # do edge detection
    #processed_image = vision.apply_edge_filter(processed_image)

    # Detect objects in the image
    boxes, scores, class_ids = yolov11_detector.detect(screenshot_masked)
    # Draw prediction boxes to original screenshot
    combined_img = yolov11_detector.draw_detections(screenshot, boxes, scores, class_ids)
    # Calculate FPS
    fps='{:.0f} fps'.format(1 / (time.time() - loop_time))
    # Draw FPS to image
    cv2.putText(combined_img, fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    # Swap RB channels back
    combined_img= cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
    # Show output
    cv2.imshow('output',combined_img)
    
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
