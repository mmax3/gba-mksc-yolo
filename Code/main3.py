import cv2 as cv2
import numpy as np
import os
import time
import subprocess
#import pyautogui
#from PIL import Image
from windowcapture import WindowCapture
from yolov11.yolocore import YOLODetector

# Initialize YOLOv11 object detector
model_path = 'models/yolov11n-mm.onnx'
yolov11_detector =  YOLODetector(model_path= model_path , conf_thresh=0.1, iou_thresh=0.45)

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

    screenshot = np.array(screenshot_raw)
    if (mask.shape!=screenshot.shape):
        print(f"Wrong screen size: {screenshot.shape}")
        mask = cv2.resize(mask, (screenshot.shape[1], screenshot.shape[0]), interpolation = cv2.INTER_NEAREST)
        #screenshot = cv2.resize(screenshot, (mask.shape[1], mask.shape[0]), interpolation = cv2.INTER_NEAREST)
        #print(mask.shape,screenshot.shape)
        #wincap = WindowCapture('Mario Kart - Super Circuit (Europe) [Gameboy Advance] - BizHawk')
        #screenshot_raw = wincap.get_screenshot()
        #screenshot = np.array(screenshot_raw)
    screenshot_masked=cv2.bitwise_and(screenshot,mask,mask=None)
   	
    # pre-process the image
    #apply filter
    #processed_image = vision.apply_hsv_filter(screenshot)
    # do edge detection
    #processed_image = vision.apply_edge_filter(processed_image)
    #cv2.imshow('output',screenshot)

    # Detect objects in the image
    boxes, scores, class_ids = yolov11_detector.detect(screenshot)
    #combined_img = yolov11_detector.draw_detections(screenshot, boxes, scores, class_ids)

    #fps='{:.0f} fps'.format(1 / (time.time() - loop_time))
    #cv2.putText(combined_img, fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    #combined_img= cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
    #cv2.imshow('output',combined_img)
    
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
