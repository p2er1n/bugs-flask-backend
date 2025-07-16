from flask import Blueprint, request, jsonify
from utils import download_file
from ortEngine import run as ort_run
import os

api_bp = Blueprint('api', __name__)

# 获取项目的根目录
ROOT  = os.path.dirname(os.path.abspath(__file__)) + "/../"

@api_bp.route('/detect', methods=['POST'])
def detect():
    req = request.get_json()
    image_url = req.get("image_url", None)
    engines = req.get("engines", "onnxruntime")
    model = req.get("model", None)

    output = None
    image = download_file(image_url)
    if engines == 'onnxruntime':
        model_path = ROOT + "weights/"+model+".onnx"
        output = ort_run(model_path, image)
    elif False:
        pass
    else:
        pass
    
    return jsonify([{
        "p1": [o["x1"], o["y1"]],
        "p2": [o["x2"], o["y2"]],
        "cls": o["cls"],
        "conf": o["conf"]
    } for o in output])