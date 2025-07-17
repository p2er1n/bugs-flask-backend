from flask import Blueprint, request, jsonify
from utils import download_file
from tasks import run_task

import os

api_bp = Blueprint('api', __name__)

# 获取项目的根目录
ROOT  = os.path.dirname(os.path.abspath(__file__)) + "/../"

@api_bp.route('/detect', methods=['POST'])
def detect():
    req = request.get_json()
    image_url = req.get("image_url", None)
    task = req.get("task", "detect")
    engine = req.get("engine", "default")
    model = req.get("model", "default")

    image = download_file(image_url)

    output = run_task(task, image, engine=engine, model=model)

    return jsonify(output)