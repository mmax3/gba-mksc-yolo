import cv2 as cv2
import numpy as np
import os
import sys
import time
import subprocess
from windowcapture import WindowCapture
from yolo26.yolo26 import YOLODetector

# Add the Code directory to Python path to ensure imports work
_code_dir = os.path.dirname(os.path.abspath(__file__))
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

print("Yolo26 ONNX with framegrabber, custom code inference")

# Initialize YOLO26 object detector
model_path = os.path.join(_code_dir, 'Models/yolo26n.onnx')
print(f"Loading YOLO model from: {model_path}")
try:
    yolo26_detector =  YOLODetector(model_path= model_path , conf_thresh=0.35, iou_thresh=0.6)
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    raise

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#WindowCapture.list_window_names()
# initialize the WindowCapture class
print("Looking for BizHawk window...")
try:
    wincap = WindowCapture('Mario Kart - Super Circuit (Europe) [Gameboy Advance] - BizHawk')
    print("BizHawk window found!")
except Exception as e:
    print(f"BizHawk window not found: {e}")
    wincap=False

if (wincap==False):
    print("Launching BizHawk emulator...")
    try:
        p=subprocess.Popen([r'..\\BizHawk-2.9.1-win-x64\\EmuHawk.exe',
                            r'..\\BizHawk-2.9.1-win-x64\\ROMS\\Mario Kart - Super Circuit.gba',
                            '--load-slot=1'
                            ], )
        print("BizHawk launched, waiting for window to appear...")
    except Exception as e:
        print(f"Error launching BizHawk: {e}")
        print("Please make sure BizHawk is installed at the expected path")
        raise

max_wait_time = 30  # Maximum 30 seconds to wait for window
wait_start = time.time()
while (wincap==False):
    try:
        wincap = WindowCapture('Mario Kart - Super Circuit (Europe) [Gameboy Advance] - BizHawk')
        print("BizHawk window found!")
        break
    except Exception as e:
        if time.time() - wait_start > max_wait_time:
            print(f"Timeout: Could not find BizHawk window after {max_wait_time} seconds")
            raise Exception("BizHawk window not found")
        time.sleep(1.0)
        continue

print("Loading mask image...")
mask = cv2.imread('mask4.jpg')
if mask is None:
    print(f"ERROR: Could not load mask image 'mask4.jpg'. Please check if the file exists.")
    raise FileNotFoundError("mask4.jpg not found")
print(f"Mask loaded successfully: {mask.shape}")

# Cache for resized mask to avoid resizing every frame
_cached_mask = mask
_cached_mask_shape = mask.shape

print("Starting main loop...")
loop_time = time.time()
frame_count = 0

while(True):
    frame_count += 1
    if frame_count == 1:
        print("First frame captured, entering main loop...")

    # get an updated image of the game
    try:
        screenshot_raw = wincap.get_screenshot()
        screenshot = np.array(screenshot_raw)# už má prehodené RB kanály
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        time.sleep(0.1)
        continue
    if (mask.shape != screenshot.shape):
        print(f"Screen size changed: {screenshot.shape}")
        # Use cached mask if already resized for this size
        if _cached_mask_shape != (screenshot.shape[0], screenshot.shape[1]):
            _cached_mask = cv2.resize(mask, (screenshot.shape[1], screenshot.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            _cached_mask_shape = (screenshot.shape[0], screenshot.shape[1])
    else:
        _cached_mask = mask
    
    # apply mask
    screenshot_masked = cv2.bitwise_and(screenshot, _cached_mask, mask=None)
    #cv2.imshow('input',screenshot_masked)
    
    # pre-process the image
    #apply filter
    #processed_image = vision.apply_hsv_filter(screenshot)
    # do edge detection
    #processed_image = vision.apply_edge_filter(processed_image)

    # Detect objects in the image
    boxes, scores, class_ids = yolo26_detector.detect(screenshot_masked)
    # Draw prediction boxes to original screenshot
    combined_img = yolo26_detector.draw_detections(screenshot, boxes, scores, class_ids)
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
    key = cv2.waitKey(1)
    if key == ord('f'):
        pass  # Debug hook
    elif key == ord('q'):
        print("Quitting...")
        cv2.destroyAllWindows()
        if 'p' in locals():
            p.terminate()
        break

print('Done.')
