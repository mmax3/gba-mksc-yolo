This project tries the object recognition YOLO models to recognize objects in Mario Cart: Super Circuit in real time

[Video from using Yolov11n:](https://youtu.be/ESSt6djRz0I)


# Installation Guide

## Requirements

python 3 or conda environment with python 3

Check the `requirements.txt` file. For ONNX, if you have an NVIDIA GPU, install `onnxruntime-gpu`; otherwise, use the `onnxruntime` library.

## Installation Steps

### Step 1: Set Up the Directory Structure

1. Create a directory of your choice, e.g., `AI-ML-Playground`.
2. Download and extract [BizHawk 2.9.1](https://tasvideos.org/Bizhawk/ReleaseHistory#Bizhawk291) into a new subdirectory `BizHawk-2.9.1-win-x64`.
3. Place `Mario Kart - Super Circuit.gba` ROM file into `BizHawk-2.9.1-win-x64/ROMs` directory.
4. Navigate to the `AI-ML-Playground` directory.
5. Manually copy all contents from this GitHub repository into `AI-ML-Playground` directory, and overwrite some BizHawk files if necessary

### Step 2: Set Up the Environment

#### If using (mini)conda environment:

1. Launch the (mini)conda prompt.
2. Navigate inside `AI-ML-Playground` directory.
3. Create a new empty virtual environment in a subdirectory `venv`, as defined in the YAML file:
   ```sh
   conda env create --prefix=venv -f environment.yml
   ```
4. To activate:
   - If inside `AI-ML-Playground` directory:
     ```sh
     activate ./venv
     ```
   - If outside directory:
     ```sh
     activate D:/AI-ML-Playground/venv
     ```

#### If using just python with pip:

```sh
pip install -r requirements.txt
```

## Configuration Notes

- `windowcapture.py`

  - Lines 40, 41, and 42 may need to be adjusted on your PC.
  These represents size of attributes of BizHawk application window, that need to be ignored by screen grabbing

- `main.py`

  - Uses a custom-trained YOLOv7-tiny model and custom inference code using `onnxruntime`.
  - Uses `win32gui` for screen grabbing.
  - Commented lines allow switching to the YOLO-NAS model.
  - Opens a filter/image enhancement window for testing; leave the sliders as they are.
  - Inference based on: [ONNX YOLOv7 Object Detection](https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection/tree/main).
  - Image enhancement based on: [OpenCV Tutorials](https://github.com/learncodebygaming/opencv_tutorials).

- `main2.py`

  - Uses a custom-trained YOLOv11n model with dynamic input and Ultralytics inference library.
  - No image enhancement window.

- `main3.py`

  - Uses a custom-trained YOLOv11n model and custom inference code using `onnxruntime`.
  - Fastest so far.
  - Inference code based on: [YOLOv11-ONNX Object Detection](https://github.com/SihabSahariar/Yolov11-ONNX-Object-Detection).

- `main4.py`

  - Uses OpenCV (`cv2`) for inference.
  - Based on guide: [Using YOLOv8 in ONNX Format](https://medium.com/@zain.18j2000/how-to-use-custom-or-official-yolov8-object-detection-model-in-onnx-format-ca8f055643df).
  
- `main-lua.py`

  - Screenshots and results are sent/received to Python script via Lua server to/from BizHawk.
  - Lua server code modified from: [BrainHawk](https://github.com/TylerLandowski/BrainHawk).
  - It also automatically executes lua script in lua console window in BizHawk, unfortunately the lua script will not run properly, you need to stop it by double-click and run it again with double-click

All scripts automatically launch Bizhawk if its not already running

## Additional Recommendations

- When BizHawk is running, set the window size to 2x for better performance.

## Resources

- **YOLO Training Scripts:** [Roboflow Notebooks](https://github.com/roboflow/notebooks)
- **Used Datasets:** [MKSC Dataset on Roboflow Universe](https://universe.roboflow.com/mmax/mksc/browse)



