from flask import Blueprint, request, jsonify
import requests
from models import User, db
import os
from utils import success, error
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, Picture, DetectionHistory
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/wechat/login', methods=['POST'])
def wechat_login():
    data = request.get_json()
    code = data.get('code')

    if not code:
        return error('缺少code参数', 400)

    WX_APPID = os.getenv("WX_APPID")
    WX_SECRET = os.getenv("WX_SECRET")

    url = f'https://api.weixin.qq.com/sns/jscode2session?appid={WX_APPID}&secret={WX_SECRET}&js_code={code}&grant_type=authorization_code'
    resp = requests.get(url).json()

    openid = resp.get('openid')

    if not openid:
        return error('微信认证失败', 400)

    user = User.get_by_openid(openid)
    if not user:
        user = User(openid=openid)
        user.save()

    token = user.get_token()
    return success({'token': token, 'user_id': user.id})

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return error('缺少用户名或密码', 400)

    if User.get_by_username(username):
        return error('用户名已存在', 400)

    new_user = User(
        username=username,
        password=password,  # 实际应加密存储：在真实项目中请使用 hash
    )
    db.session.add(new_user)
    db.session.commit()
    token = new_user.get_token()
    return success({'token': token, 'user_id': new_user.id})

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.get_by_username(username)

    if not user or user.password != password:
        return error('用户名或密码错误', 401)

    token = user.get_token()
    return success({'token': token, 'user_id': user.id, 'username': user.username, 'avatar_url': user.avatar_url})

@auth_bp.route('/history', methods=['GET'])
@jwt_required()
def get_history():
    current_user_id = get_jwt_identity()
    histories = DetectionHistory.query.filter_by(user_id=current_user_id).all()
    return success([h.to_dict() for h in histories])