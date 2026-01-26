import cv2 as cv2
import numpy as np
import os
import time
import sys
import subprocess
from windowcapture import WindowCapture
from yolov11.yolocore import YOLODetector
from BHServer import BHServer

# Add the Code directory to Python path to ensure imports work
_code_dir = os.path.dirname(os.path.abspath(__file__))
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

print("YoloV11 ONNX with Lua server, HTTP screenshot transfer")
print("Uses BizHawk's comm.httpPostScreenshot() for screenshot capture")
print("Server stores and provides screenshots to YOLO detector")
print("If it doesn't run restart LUA script")

# Initialize YOLOv11 object detector
model_path = os.path.join(_code_dir, 'Models/yolov11n.onnx')
yolov11_detector =  YOLODetector(model_path= model_path , conf_thresh=0.35, iou_thresh=0.6)
# Change the working directory to the folder this script is in.

# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

mask1 = cv2.imread('mask4.jpg')
_cached_mask = None  # Cache for resized mask
_cached_mask_shape = None

# Start the TCP server
server = BHServer(
    # Server Settings
    ip = "127.0.0.1",
    port = 1337,
    # Data Settings
    use_grayscale = False,  # Store screenshots in grayscale
    system = "GBA",  # Initialize server.controls to standard N64 controls
    # Client Settings
    mode = "HUMAN",
    update_interval = 1,  # Update to server every 5 frames
    frameskip = 0,
    speed = 100,  # Emulate at 6399% original game speed (max)
    sound = False,  # Turn off sound
    rom = "ROMs/Mario Kart - Super Circuit.gba",  # Add a game ROM file
    saves = {"GBA/State/Mario Kart - Super Circuit (Europe).mGBA.QuickSave1.State": 100}  # Add a save state
)
server.start()

def update(self):
    """Update game state and run YOLO detection."""
    actions = self.actions  # Grab number of times update() has been called
    
    # Safely access the latest screenshot
    if actions - 1 not in self.screenshots:
        print(f"WARNING: Screenshot not available for action {actions - 1}")
        return
    
    ss = self.screenshots[actions - 1]  # Latest screenshot (numpy.ndarray)
    
    # Scale screenshot if available from server
    if ss.shape != (0,):
        screenshot = scale_2x(ss)
    else:
        # Fallback: capture from window
        screenshot_raw = wincap.get_screenshot()
        screenshot = np.array(screenshot_raw)
    
    # Get or resize mask (cached for performance)
    global _cached_mask, _cached_mask_shape, mask1
    target_shape = (screenshot.shape[0], screenshot.shape[1])
    
    if _cached_mask_shape != target_shape:
        if mask1.shape != (screenshot.shape[0], screenshot.shape[1]):
            _cached_mask = cv2.resize(mask1, (screenshot.shape[1], screenshot.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        else:
            _cached_mask = mask1
        _cached_mask_shape = target_shape
    
    screenshot_masked = cv2.bitwise_and(screenshot, _cached_mask, mask=None)
    
    # Run YOLO detection
    boxes, scores, class_ids = yolov11_detector.detect(screenshot_masked)
    
    # Format detection results efficiently (vectorized)
    if len(class_ids) > 0:
        # Convert scores to percentages as integers
        scores_int = (np.array(scores) * 100).astype(int)
        # Create 1-indexed strings
        class_ids_string = ",".join(f"{i+1}:{cid}" for i, cid in enumerate(class_ids))
        scores_string = ",".join(f"{i+1}:{s}" for i, s in enumerate(scores_int))
    else:
        class_ids_string = ""
        scores_string = ""
    
    # Send detection results back to Lua
    self.data = {
        "boxes": ("STRING[][]", [[int(coord/2) for coord in coordinates] for coordinates in boxes]),
        "scores": ("INT[]", scores_string),
        "class_ids": ("INT[]", class_ids_string)
    }
    

def scale_2x(original_image):
    """Scale image 2x using nearest neighbor interpolation.
    
    Args:
        original_image: Input image (uint8 or other dtype)
        
    Returns:
        Scaled image as uint8
    """
    # Convert to uint8 if needed
    if original_image.dtype != np.uint8:
        original_image = cv2.normalize(original_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Scale 2x with nearest neighbor
    scaled = cv2.resize(original_image, 
                       (original_image.shape[1] * 2, original_image.shape[0] * 2),
                       interpolation=cv2.INTER_NEAREST)
    return scaled

# Replace the server's update function with ours
BHServer.update = update
print(f"Server ready at IP:{server.ip} port:{server.port}")
print(f"Run EmuHawk.exe with these parameters:")
print(f"--socket_ip={server.ip} --socket_port={server.port} --url_get=http://{server.ip}:9876/get --url_post=http://{server.ip}:9876/post")

#WindowCapture.list_window_names()
# initialize the WindowCapture class
try:
    wincap = WindowCapture('Mario Kart - Super Circuit (Europe) [Gameboy Advance] - BizHawk')
except:
    wincap=False

if (wincap==False):
    p=subprocess.Popen([r'..\\BizHawk-2.9.1-win-x64\\EmuHawk.exe', #An 'r' before a string tells the Python interpreter to treat backslashes as a literal (raw) character. Normally, Python uses backslashes as escape characters
                        #'..\BizHawk-2.9.1-win-x64\ROMS\Mario Kart - Super Circuit.gba', # handled by BHServer and luascript
                        #'--load-slot=1', # handled by BHServer and luascript
                        r'--lua=..\\BizHawk-2.9.1-win-x64\\Lua\\BrainHawk-MM\\SampleTool.lua',
                        f'--socket_ip={server.ip}',
                        f'--socket_port={server.port}',
                        f'--url_get=http://{server.ip}:9876/get',
                        f'--url_post=http://{server.ip}:9876/post'
                        ]
                       )

while (wincap==False):
    try:
        wincap = WindowCapture('Mario Kart - Super Circuit (Europe) [Gameboy Advance] - BizHawk')
    except:
        time.sleep(1.0)
        continue

while(True):
    # Press 'q' to exit, 'f' for debug (calls waitKey once per loop)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('f'):
        pass  # Debug hook for future use

print('Done.')
