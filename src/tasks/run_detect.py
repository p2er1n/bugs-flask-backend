import os
import onnxruntime as ort
import ncnn
import numpy as np
import cv2
from .utils import CLASSES, filter_Detections, rescale_back, WEIGHTS_ROOT

DETECT_DEFAULT_MODEL = "yolov5nu"
DETECT_DEFAULT_ENGINE = "onnxruntime"

engine = None

def init_ort_engine(model):
    global engine
    engine = ort.InferenceSession(f"{WEIGHTS_ROOT}{model}.onnx")

def run_ort_engine(ipt):
    global engine
    ipt_name = engine.get_inputs()[0].name
    opt = engine.run(None, {ipt_name: ipt})
    return opt[0][0].transpose()

def init_ncnn_engine(model):
    global engine
    engine = ncnn.Net()
    engine.load_param(f"{WEIGHTS_ROOT}{model}_ncnn/model.ncnn.param")
    engine.load_model(f"{WEIGHTS_ROOT}{model}_ncnn/model.ncnn.bin")

def run_ncnn_engine(ipt):
    global engine
    outputs = []
    with engine.create_extractor() as ex:
        #ex.input("in0", ncnn.Mat(np.squeeze(img, axis=0)).clone())
        ex.input(engine.input_names()[0], ipt)

        # _, out0 = ex.extract("out0")
        # outputs.append(np.expand_dims(np.array(out0), axis=0))
        outputs = [np.array(ex.extract(x)[1])[None] for x in sorted(engine.output_names())]
    return outputs[0][0].transpose()

def yolo_ncnn_preprocess(image):
    np_image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR) 
    img_w, img_h = image.shape[1], image.shape[0]
    w = img_w
    h = img_h
    scale = 1.0
    if w > h:
        scale = float(640) / w
        w = 640
        h = int(h * scale)
    else:
        scale = float(640) / h
        h = 640
        w = int(w * scale)

    mat_in = ncnn.Mat.from_pixels_resize(
        image, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h, w, h
    )
    # pad to target_size rectangle
    # yolov5/utils/datasets.py letterbox
    wpad = (w + 31) // 32 * 32 - w
    hpad = (h + 31) // 32 * 32 - h
    mat_in_pad = ncnn.copy_make_border(
        mat_in,
        hpad // 2,
        hpad - hpad // 2,
        wpad // 2,
        wpad - wpad // 2,
        ncnn.BorderType.BORDER_CONSTANT,
        114.0,
    )

    mat_in_pad.substract_mean_normalize([], [1 / 255.0, 1 / 255.0, 1 / 255.0])
    mat_in = ncnn.Mat(mat_in_pad)
    return mat_in, {
        "img_w": img_w,
        "img_h": img_h,
    }


def yolo_preprocess(image):
    np_image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR) 
    img_w, img_h = image.shape[1], image.shape[0]
    img = cv2.resize(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, 640, 640)
    img = img / 255.0
    img = img.astype(np.float32)

    return img, {
        "img_w": img_w,
        "img_h": img_h,
    }

def yolo_postprocess(opt, state):
    img_w = state["img_w"]
    img_h = state["img_h"]

    print(opt)

    results = filter_Detections(opt)
    rescaled_results, confidences = rescale_back(results, img_w, img_h)

    output = []
    for res, conf in zip(rescaled_results, confidences):
        x1,y1,x2,y2, cls_id = res
        cls_id = int(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = "{:.2f}".format(conf)
        output.append({ "box":{
                "x": x1,
                "y": y1,
                "width": x2-x1,
                "height": y2 - y1            
                },
            "className": CLASSES[cls_id],
            "confidence": conf
        })
    print(output)
    return output

def rtdetr_preprocess(image):
    np_image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR) 
    img_w, img_h = image.shape[1], image.shape[0]
    img = cv2.resize(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, 640, 640)
    img = img / 255.0
    img = img.astype(np.float32)

    return img, {
        "img_w": img_w,
        "img_h": img_h,
    }

def rtdetr_postprocess(output, state, confidence_threshold=0.3):
    output = output.transpose()
    # Get the bounding box and class scores
    bboxes = output[:, :4]  # First 4 values: bounding box [cx, cy, width, height]
    logits = output[:, 4:]

    # Get the best class index and confidence score for each object
    best_class_indices = np.argmax(logits, axis=-1)
    best_class_confidences = np.max(logits, axis=-1)

    # Filter detections based on the confidence threshold
    high_confidence_indices = best_class_confidences > confidence_threshold
    best_class_indices = best_class_indices[high_confidence_indices]
    best_class_confidences = best_class_confidences[high_confidence_indices]
    bboxes = bboxes[high_confidence_indices]

    # Convert bounding boxes from [cx, cy, w, h] to [x_min, y_min, x_max, y_max]
    cx, cy, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x_min = (cx - 0.5 * w)
    y_min = (cy - 0.5 * h)
    x_max = (cx + 0.5 * w)
    y_max = (cy + 0.5 * h)

    # Prepare the final list of detections using your desired format
    detections = []
    for i in range(len(best_class_indices)):
        conf = "{:.2f}".format(float(best_class_confidences[i]))  # Format confidence as string with two decimals
        cls_id = best_class_indices[i]
    
        detection = {
            "box": {
                "x": float(x_min[i]),
                "y": float(y_min[i]),
                "width": float(x_max[i] - x_min[i]),
                "height": float(y_max[i] - y_min[i])
            },
            "className": str(CLASSES[cls_id]),  # Assuming CLASSES is a list of class names
            "confidence": conf  # As string
        }
        detections.append(detection)
    
    print(detections)
    return detections

    # # Prepare the final list of detections
    # detections = []
    # for i in range(len(best_class_indices)):
    #     detection = {
    #         "class": int(best_class_indices[i]),
    #         "confidence": float(best_class_confidences[i]),
    #         "bbox": [float(x_min[i]), float(y_min[i]), float(x_max[i]), float(y_max[i])]
    #     }
    #     detections.append(detection)

    # return detections


def run_detect(image, engine, model):
    if engine == "default":
        engine = DETECT_DEFAULT_ENGINE
    if model == "default":
        model = DETECT_DEFAULT_MODEL
    
    if engine == "onnxruntime":
        init_ort_engine(model)
    elif engine == "ncnn":
        init_ncnn_engine(model)

    ipt = None
    opt = None

    if model[:4] == "yolo":
        if engine == "ncnn":
            ipt, state = yolo_ncnn_preprocess(image)
        else:
            ipt, state = yolo_preprocess(image)
    elif model[:6] == "rtdetr":
        ipt, state = rtdetr_preprocess(image)

    if engine == "onnxruntime":
        opt = run_ort_engine(ipt)
    elif engine == "ncnn":
        opt = run_ncnn_engine(ipt)
    
    if model[:4] == "yolo":
        opt = yolo_postprocess(opt, state)
    elif model[:6] == "rtdetr":
        opt = rtdetr_postprocess(opt, state)

    return opt
    