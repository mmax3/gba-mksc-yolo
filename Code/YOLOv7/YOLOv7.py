import time
import cv2
import numpy as np
import onnxruntime
from .utils import draw_detections, nms#,xywh2xyxy 
import argparse

class YOLOv7:

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
        # Get model info
        self.get_input_details()
        self.get_output_details()

        self.has_postprocess = 'score' in self.output_names

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)
        #print(outputs)
        
        if self.has_postprocess:
            self.boxes, self.scores, self.class_ids = self.parse_processed_output(outputs)

        else:
            # Process output data
            self.boxes, self.scores, self.class_ids = self.process_output(outputs)
         
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = img.copy()
        image, self.ratio, self.dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        im.shape
        return im

    def inference(self, input_tensor):
        #start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        #print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        
        ooo=output[0]
        #print(ooo.ndim)
        if (ooo.ndim==1):
            ooo=[[ooo]]#np.expand_dim(ooo,axis=0)
        elif (ooo.ndim==0):
            ooo=[[ooo]]#np.expand_dim(np.expand_dim(ooo,axis=0))
        predictions = ooo    
        
        #predictions = np.squeeze(output[0])
        #print(predictions.ndim)
        #batch_id,x0,y0,x1,y1,cls_id,score = zip(*output)

        #netuším, čo chcel tým povedať
        # Filter out object confidence scores below threshold
        #obj_conf = predictions[:, 4]
        #predictions = predictions[obj_conf > self.conf_threshold]
        #obj_conf = obj_conf[obj_conf > self.conf_threshold]

        #netuším, čo chcel tým povedať
        # Multiply class confidence with bounding box confidence
        #predictions[:, 5:] *= obj_conf[:, np.newaxis]
        
        # Get the scores
        #scores = np.max(predictions[:, 5:], axis=1)
        scores = np.squeeze(predictions[:, 6:],axis=1)
        #scores=np.array(score)
 
        # Filter out the objects with a low score
        predictions = predictions[scores > self.conf_threshold]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []
        
        # Get the class with the highest confidence
        #class_ids = np.argmax(predictions[:, 5:], axis=1)
        class_ids = np.squeeze(predictions[:, 5:6].astype(int),axis=1)
        #class_ids=np.array(cls_id,dtype=int)
        
        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)
        #indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold).flatten()#nefunguje, niekedy dá [0] namiesto [0 1 2]
        return boxes[indices], scores[indices], class_ids[indices]

    def parse_processed_output(self, outputs):

        scores = np.squeeze(outputs[0], axis=1)
        predictions = outputs[1]
        # Filter out object scores below threshold
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]

        if len(scores) == 0:
            return [], [], []

        # Extract the boxes and class ids
        # TODO: Separate based on batch number
        batch_number = predictions[:, 0]
        class_ids = predictions[:, 1]
        boxes = predictions[:, 2:]

        # In postprocess, the x,y are the y,x
        boxes = boxes[:, [1, 0, 3, 2]]

        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return boxes, scores, class_ids

    #def extract_boxes(self, predictions):
    def extract_boxes(self, predictions):
       
        # Extract boxes from predictions
        #boxes = predictions[:, :4]
        boxes = predictions[:, 1:5]
        #print(boxes)
        #x0=np.array(x0)
        #y0=np.array(y0)
        #x1=np.array(x1)
        #y1=np.array(y1)
        #box = np.array([x0,y0,x1,y1]).T
        ##box -= np.array(self.dwdh*2)
        ##box /= self.ratio
        ##box = box.round().astype(np.int32).tolist()
        #print(box)

        # Convert boxes to xyxy format
        ##boxes = xywh2xyxy(boxes)
        
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)
        #print(self.input_width, self.input_height)
        #print(self.img_width, self.img_height)

        return boxes

    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        #print(self.dwdh)
        #print(self.ratio)
        boxes -= np.array(self.dwdh*2)
        #boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes = np.divide(boxes, self.ratio, dtype=np.float32)
        #boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores, self.class_ids, mask_alpha)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def letterbox(self, im, new_shape=(480, 320), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
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
    yolov7_detector = YOLOv7(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)
    srcimg = cv2.imread(args.imgpath)

    # Detect Objects
    boxes, scores, class_ids = yolov7_detector.detect(srcimg)
    
    # Draw detections
    dstimg = yolov7_detector.draw_detections(srcimg, boxes, scores, class_ids)
    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
