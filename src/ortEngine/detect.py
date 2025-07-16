import onnxruntime as ort
import cv2
import numpy as np
from .utils import CLASSES, filter_Detections, rescale_back

def run(model_path, image):
    onnx_model = ort.InferenceSession(model_path)

    np_image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR) 
    img_w, img_h = image.shape[1], image.shape[0]

    img = cv2.resize(image, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, 640, 640)

    # Normalize pixel values to the range [0, 1]
    img = img / 255.0

    # Convert image to float32
    img = img.astype(np.float32)

    outputs = onnx_model.run(None, {"images": img})
    results = outputs[0][0].transpose()

    results = filter_Detections(results)

    rescaled_results, confidences = rescale_back(results, img_w, img_h)

    output = []

    for res, conf in zip(rescaled_results, confidences):

        x1,y1,x2,y2, cls_id = res
        cls_id = int(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = "{:.2f}".format(conf)

        output.append({'x1': x1, 'y1':y1, 'x2':x2, 'y2':y2,"cls": CLASSES[cls_id],"conf": conf})

    return output