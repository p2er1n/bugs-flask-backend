from .utils import WEIGHTS_ROOT, postprocess_obb, CLASSES
import onnxruntime as ort
import numpy as np
from ultralytics import YOLO
import cv2

DETECT_DEFAULT_MODEL = "yolo11n-obb"
DETECT_DEFAULT_ENGINE = "ultralytics"

engine = None

# def init_ort_engine(model):
#     global engine
#     engine = ort.InferenceSession(f"{WEIGHTS_ROOT}{model}.onnx")

# def run_ort_engine(ipt):
#     global engine
#     ipt_name = engine.get_inputs()[0].name
#     opt = engine.run(None, {ipt_name: ipt})
#     return opt[0][0].transpose()

# def yolo_preprocess(image):
#     np_image = np.frombuffer(image, dtype=np.uint8)
#     image = cv2.imdecode(np_image, cv2.IMREAD_COLOR) 
#     img_w, img_h = image.shape[1], image.shape[0]
#     img = cv2.resize(image, (1024, 1024))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.transpose(2, 0, 1)
#     img = img.reshape(1, 3, 1024, 1024)
#     img = img / 255.0
#     img = img.astype(np.float32)

#     return img, {
#         "img_w": img_w,
#         "img_h": img_h,
#     }

# def yolo_postprocess(opt, state):
#     return postprocess_obb(opt, (state["img_w"], state["img_h"]))

def init_ultralytics_engine(model):
    global engine
    formatt = "onnx"
    if engine is not None:
        return
    print(123)
    engine = YOLO(f"{WEIGHTS_ROOT}{model}.{formatt}", task="obb")

def run_ultralytics_engine(ipt):
    opt = engine.predict(ipt, device='cpu')
    return opt

def yolo_ultralytics_preprocess(image):
    np_image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR) 
    return image

def yolo_ultralytics_postprocess(opt):
    results = []
    # opt is a list of Results objects, we process the first one for the single image.
    if not opt:
        return []
        
    detections = opt[0].obb  # Get the OBB object with all detections
    for i in range(len(detections.xyxyxyxy)):
        xyxyxyxy = detections.xyxyxyxy[i]
        conf = detections.conf[i]
        cls_id = detections.cls[i]
        x1 = int(xyxyxyxy[0][0])
        y1 = int(xyxyxyxy[0][1])
        x2 = int(xyxyxyxy[1][0])
        y2 = int(xyxyxyxy[1][1])
        x3 = int(xyxyxyxy[2][0])
        y3 = int(xyxyxyxy[2][1])
        x4 = int(xyxyxyxy[3][0])
        y4 = int(xyxyxyxy[3][1])

        results.append({
            "box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "x3": x3,
                "y3": y3,
                "x4": x4,
                "y4": y4,
            },
            "confidence": "{:.2f}".format(conf),
            "className": CLASSES[int(cls_id)],
        })
    return results

def run_obb(image, engine, model):
    if engine == "default":
        engine = DETECT_DEFAULT_ENGINE
    if model == "default":
        model = DETECT_DEFAULT_MODEL
    
    if engine == "onnxruntime":
        # init_ort_engine(model)
        pass
    elif engine == "ultralytics":
        init_ultralytics_engine(model)
    else:
        pass

    ipt = None
    opt = None

    if model[:4] == "yolo":
        if engine == "ultralytics":
            ipt = yolo_ultralytics_preprocess(image)
        else:
            # ipt, state = yolo_preprocess(image)
            pass
    elif False:
        pass

    if engine == "onnxruntime":
        # opt = run_ort_engine(ipt)
        pass
    elif engine == 'ultralytics':
        opt = run_ultralytics_engine(ipt)
    else:
        pass
    if model[:4] == "yolo":
        if engine == "ultralytics":
            opt = yolo_ultralytics_postprocess(opt)
        else:
            # opt = yolo_postprocess(opt, state)
            pass
    elif False:
        pass

    return opt
    