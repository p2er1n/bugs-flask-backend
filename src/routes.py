from flask import Blueprint, request, jsonify, current_app, Response
from flask_jwt_extended import jwt_required, get_jwt_identity
from tasks import run_task
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

# @api_bp.route('/detect', methods=['POST'])
# def detect():
#     req = request.get_json()
#     image_url = req.get("image_url", None)
#     task = req.get("task", "detect")
#     engine = req.get("engine", "default")
#     model = req.get("model", "default")

#     # image = download_file(image_url)

#     output = run_task(task, image, engine=engine, model=model)

#     return jsonify(output)
    
@api_bp.route('/identify', methods=['POST'])
@jwt_required()
def upload_and_identify():
    if 'file' not in request.files:
        return error("No file part", 400)
    file = request.files['file']
    if file.filename == '':
        return error("No selected file", 400)
    
    task = request.form.get("task", "obb")
    # task = request.form.get("task", "detect")
    engine = request.form.get("engine", "default")
    model = request.form.get("model", "default")

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

        detections = run_task(task, file_bytes, engine=engine, model=model)
        detection_result_json = json.dumps(detections)
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
            "detections": detections
        })

    except Exception as e:
        return error(f"An error occurred: {str(e)}", 500)


@api_bp.route('/realtime_identify', methods=['POST'])
@jwt_required()
def realtime_identify():
    if 'file' not in request.files:
        return error("No file part", 400)
    file = request.files['file']
    if file.filename == '':
        return error("No selected file", 400)
    
    task = request.form.get("task", "obb")
    engine = request.form.get("engine", "default")
    model = request.form.get("model", "default")

    try:
        file_bytes = file.read()
        detections = run_task(task, file_bytes, engine=engine, model=model)
        
        return success({
            "detections": detections
        })

    except Exception as e:
        return error(f"An error occurred: {str(e)}", 500)

