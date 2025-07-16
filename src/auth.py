from flask import Blueprint, request, jsonify
import requests
from models import User, db
import os

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/wechat/login', methods=['POST'])
def wechat_login():
    data = request.get_json()
    code = data.get('code')

    if not code:
        return jsonify({'msg': '缺少code参数'}), 400

    WX_APPID = os.getenv("WX_APPID")
    WX_SECRET = os.getenv("WX_SECRET")

    url = f'https://api.weixin.qq.com/sns/jscode2session?appid={WX_APPID}&secret={WX_SECRET}&js_code={code}&grant_type=authorization_code'
    resp = requests.get(url).json()

    openid = resp.get('openid')

    if not openid:
        return jsonify({'msg': '微信认证失败'}), 400

    user = User.get_by_openid(openid)
    if not user:
        user = User(openid=openid)
        user.save()

    token = user.get_token()
    return jsonify({'token': token, 'user_id': user.id})

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    mobile = data.get('mobile')

    if not username or not password:
        return jsonify({'msg': '缺少用户名或密码'}), 400

    if User.get_by_username(username):
        return jsonify({'msg': '用户名已存在'}), 400

    new_user = User(
        username=username,
        password=password,  # 实际应加密存储：在真实项目中请使用 hash
        mobile=mobile
    )
    db.session.add(new_user)
    db.session.commit()
    token = new_user.get_token()
    return jsonify({'token': token, 'user_id': new_user.id})

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.get_by_username(username)

    if not user or user.password != password:
        return jsonify({'msg': '用户名或密码错误'}), 401

    token = user.get_token()
    return jsonify({'token': token, 'user_id': user.id})
