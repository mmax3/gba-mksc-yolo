import cv2 as cv2
import numpy as np
import os
import time
import subprocess
#import pyautogui
#from PIL import Image
from windowcapture import WindowCapture
from YOLOv7 import YOLOv7
from YOLONAS import YOLONAS
from vision import Vision
from hsvfilter import HsvFilter
from edgefilter import EdgeFilter
from BHServer import BHServer

#class_names = list(map(lambda x: x.strip(), open('classes.txt', 'r').readlines()))

mask1 = cv2.imread('mask4.jpg')

loop_time = time.time()

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
    
    """
    if self.client_started():
        print(self.actions)
        print(self.screenshots[self.actions - 1].shape)
    if self.controls["B"]:        # If B button is pressed prints all input states
        pass
        #print(self.controls)
    """
    actions = self.actions              # Grab number of times update() has been called
    ss = self.screenshots[actions - 1]  # Grab the latest screenshot (numpy.ndarray)

    #self.controls["A"] = True    # Press the A button on Player 1's controller, mode has to be other than "HUMAN"
    """
    x_type = self.data["x"][0]    # Get type of variable x: "INT". Set by client
    x = self.data["x"][1]         # Get value of variable x: 512. Set by client
    print(f"type:{x_type} value:{x}")
    """
    
    """
    if actions == 20:
        self.save_screenshots(0, actions - 1, "my_screenshot")
    elif actions == 40:
        self.new_episode()      # Reset the emulator, actions = 0, ++episodes
        if self.episodes == 3:  # Stop client after 3 episodes
            self.exit_client()
    """
    global mask1
    mask = mask1
    
    if ss.shape != (0,):
        #ak už server posiela screenshoty
        screenshot = scale_2x(ss)
    else:
        #ak ešte nie tak snímame okno aplikácie
        # get an updated image of the game
        screenshot_raw = wincap.get_screenshot()
        screenshot = np.array(screenshot_raw)
    

    if (mask.shape!=screenshot.shape): #480x320
        print(f"Wrong screen size: {screenshot.shape}")
        mask = cv2.resize(mask, (screenshot.shape[1], screenshot.shape[0]), interpolation = cv2.INTER_NEAREST)

    screenshot_masked=cv2.bitwise_and(screenshot,mask,mask=None)
                
    # pre-process the image
    #apply filter
    #processed_image = vision.apply_hsv_filter(screenshot)
    # do edge detection
    #processed_image = vision.apply_edge_filter(processed_image)
 
    # Detect Objects
    boxes, scores, class_ids = yolo_detector(screenshot_masked)
    class_ids_string = ",".join(f"{i+1}:{value}" for i, value in enumerate(class_ids))
    scores_string = ",".join(f"{i+1}:{value}" for i, value in enumerate([int(score*100) for score in scores]))
 
    self.data = {
        #"y": ("INT", "42"),
        #"z": ("INT", "43"),
        "boxes": ("STRING[][]", [[int(coord/2) for coord in coordinates] for coordinates in boxes]),
        "scores": ("INT[]", scores_string),
        "class_ids": ("INT[]", class_ids_string)
    }
    
    '''
    # Draw detections
    combined_img = yolo_detector.draw_detections(processed_image)

    # for PT model
    #combined_img=yolov7_detector(screenshot)
    
    fps='{:.0f} fps'.format(1 / (time.time() - loop_time))
    cv2.putText(combined_img, fps, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    combined_img= cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('output',combined_img)
    '''
    # debug the loop rate
    #print('FPS {}'.format(1 / (time.time() - loop_time)))
    #loop_time = time.time()
    

def scale_2x(original_image):

    int8_image = cv2.normalize(original_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    scaled_image = cv2.resize(int8_image, (int8_image.shape[1]*2, int8_image.shape[0]*2), interpolation = cv2.INTER_NEAREST)
    return(scaled_image)

# Replace the server's update function with ours
BHServer.update = update
print(f"Server ready at IP:{server.ip} port:{server.port}")
print(f"Run EmuHawk.exe with these parameters:")
print(f"--socket_ip={server.ip} --socket_port={server.port} --url_get=http://{server.ip}:9876/get --url_post=http://{server.ip}:9876/post")

# Initialize YOLOv7 object detector
yolo_detector = YOLOv7("yolov7-tiny.onnx", conf_thres=0.35, iou_thres=0.65)
#yolo_detector = YOLONAS("yolo_nas_s.onnx", conf_thres=0.25, iou_thres=0.65)

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
    p=subprocess.Popen([r'..\\BizHawk-2.9.1-win-x64\EmuHawk.exe', #An 'r' before a string tells the Python interpreter to treat backslashes as a literal (raw) character. Normally, Python uses backslashes as escape characters
                        #'..\BizHawk-2.9.1-win-x64\ROMS\Mario Kart - Super Circuit.gba', # handled by BHServer and luascript
                        #'--load-slot=1', # handled by BHServer and luascript
                        r'--lua=..\BizHawk-2.9.1-win-x64\Lua\BrainHawk-MM\SampleTool.lua',
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

#nn = cv2.dnn.readNet("best.onnx") #nefunguje

# initialize the Vision class
#vision = Vision()
# initialize the trackbar window
#vision.init_control_gui()

# HSV filter
#hsv_filter = HsvFilter(0, 180, 129, 15, 229, 243, 143, 0, 67, 0)
#edge_filter = EdgeFilter(kernelSize=1, erodeIter=1, dilateIter=1, canny1=100, canny2=200)


while(True):
   
    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv2.waitKey(1) == ord('f'):
        pass
    elif cv2.waitKey(1) == ord('q'):
        break

print('Done.')
