import time
import cv2
import numpy as np
import onnxruntime
from .utils import draw_detections, nms#,xywh2xyxy 
import argparse

class YOLONAS:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.cuda = False
        #self.dwdh=1
        #self.ratio=1

        # Initialize model
        self.initialize_model(path, self.cuda)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path, cuda):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(path,providers=providers)
        
        #self.net = cv2.dnn.readNet(path)
        #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        #net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        #self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        #self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get model info
        self.get_input_details()
        self.get_output_details()

        self.has_postprocess = 'score' in self.output_names

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)
        #print(outputs)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, self.ratio, self.dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        #im /= 255
        im.shape
        return im

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        # Extract output detection
        #class_ids, confs, boxes = list(), list(), list()

        bboxes=output[0].squeeze()
        confidences=output[1].squeeze()
        
        confs=np.max(confidences,axis=1)
        class_ids=np.argmax(confidences,axis=1)
        
        class_ids=class_ids[confs>self.conf_threshold]
        bboxes=bboxes[confs>self.conf_threshold]
        confs=confs[confs>self.conf_threshold]

        if len(confs) == 0:
            return [], [], []

        r_class_ids, r_confs, r_boxes = list(), list(), list()
        #print(bboxes.shape,confidences.shape)
        #print(confs[0])
        #print(confidences[0])

        indexes = cv2.dnn.NMSBoxes(bboxes, confs, self.conf_threshold, self.iou_threshold)

        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i])
            r_boxes.append(bboxes[i])
            
        r_boxes = self.rescale_boxes(r_boxes)

        return r_boxes, r_confs, r_class_ids

### old initial version
    def process_output2(self, output):
        '''
        ooo=output[0]
        #print(ooo.ndim)
        if (ooo.ndim==1):
            ooo=[[ooo]]#np.expand_dim(ooo,axis=0)
        elif (ooo.ndim==0):
            ooo=[[ooo]]#np.expand_dim(np.expand_dim(ooo,axis=0))
        predictions = ooo    
        '''
        # Extract output detection
        class_ids, confs, boxes = list(), list(), list()

        bboxes=output[0].squeeze()
        confidences=output[1].squeeze()

        for i in range(confidences.shape[0]):
            conf = np.max(confidences[i])
            classes_score = confidences[i]
            class_id = np.argmax(confidences[i])

            if (classes_score[class_id] > self.conf_threshold):
                confs.append(conf)
                class_ids.append(class_id)
                boxes.append(bboxes[i])

        r_class_ids, r_confs, r_boxes = list(), list(), list()
        #print(bboxes.shape,confidences.shape)
        #print(confs[0])
        #print(confidences[0])

        indexes = cv2.dnn.NMSBoxes(boxes, confs, self.conf_threshold, self.iou_threshold)

        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i])
            r_boxes.append(boxes[i])
            
        r_boxes = self.rescale_boxes(r_boxes)

        return r_boxes, r_confs, r_class_ids

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        print(self.dwdh)
        print(self.ratio)
        boxes -= np.array(self.dwdh*2)
        #boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes = np.divide(boxes, self.ratio, dtype=np.float32)
        #boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores, self.class_ids, mask_alpha)
        
    def decodeBoundingBoxes(scores, geometry, scoreThresh):
        detections = []
        confidences = []

        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
        assert len(scores.shape) == 10, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
        assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):

            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # If score is lower than threshold score, move to next x
                if (score < scoreThresh):
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return detections and confidences
        return [detections, confidences]

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
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
    yolonas_detector = YOLONAS(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    srcimg = cv2.imread(args.imgpath)

    # Detect Objects
    boxes, scores, class_ids = yolov7_detector.detect(srcimg)

    # Draw detections
    dstimg = yolonas_detector.draw_detections(srcimg, boxes, scores, class_ids)
    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
