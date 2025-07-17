from flask import Blueprint, request, jsonify, current_app, Response
from flask_jwt_extended import jwt_required, get_jwt_identity
from ortEngine import run as ort_run
import os
import hashlib
from datetime import datetime
import json
from models import db, Picture, DetectionHistory
from utils import success, error

api_bp = Blueprint('api', __name__)

# 获取项目的根目录
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
# UPLOAD_FOLDER = os.path.join(ROOT, 'instance', 'uploads')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# @api_bp.route('/identify', methods=['POST'])
# # @jwt_required()
# def detect():
#     image = request.files.get("file", None)
#     if image is None:
#         return error("No image provided", 400)

#     image_bytes = image.read()
#     model_path = os.path.join(ROOT, "weights/best-v5nu.onnx")
#     output = ort_run(model_path, image_bytes)

#     return success([{
#         "p1": [o["x1"], o["y1"]],
#         "p2": [o["x2"], o["y2"]],
#         "cls": o["cls"],
#         "conf": o["conf"]
#     } for o in output])


@api_bp.route('/identify', methods=['POST'])
@jwt_required()
def upload_and_identify():
    if 'file' not in request.files:
        return error("No file part", 400)
    file = request.files['file']
    if file.filename == '':
        return error("No selected file", 400)

    try:
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        file_bytes = file.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        existing_picture = Picture.get_by_hash(file_hash)

        if existing_picture:
            picture = existing_picture
        else:
            filename = f"{file_hash}_{file.filename}"
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            with open(file_path, 'wb') as f:
                f.write(file_bytes)

            file_url = f"{current_app.config['BASE_FILE_URL']}/uploads/{filename}"

            current_user_id = get_jwt_identity()
            new_picture = Picture(
                url=file_url,
                hash=file_hash,
                user_id=current_user_id
            )
            new_picture.save()
            picture = new_picture

        model_path = os.path.join(ROOT, "weights/best-v5nu.onnx")
        detections = ort_run(model_path, file_bytes)
        detection_result = [{
            # "box":{
            #     "x": o["x1"],
            #     "y": o["y1"],
            # }
            "box":{
                "x": o["x1"],
                "y": o["y1"],
                "width": o["x2"] - o["x1"],
                "height": o["y2"] - o["y1"]
            },
            # "p1": [o["x1"], o["y1"]],
            # "p2": [o["x2"], o["y2"]],
            "className": o["cls"],
            "confidence": o["conf"]
        } for o in detections]
        detection_result_json = json.dumps(detection_result)

        current_user_id = get_jwt_identity()
        history = DetectionHistory(
            user_id=current_user_id,
            picture_id=picture.id,
            result=detection_result_json,
            detection_time=datetime.utcnow()
        )
        db.session.add(history)
        db.session.commit()
        return success({
            "fileUrl": picture.url,
            "detections": detection_result
        })

    except Exception as e:
        return error(f"An error occurred: {str(e)}", 500)

