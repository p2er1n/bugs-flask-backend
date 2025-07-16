from flask import Flask, jsonify
from flask_jwt_extended import JWTManager
from config import Config
from models import db
from auth import auth_bp
from routes import api_bp

app = Flask(__name__)
app.config.from_object(Config)

# flask扩展初始化
jwt = JWTManager(app) # jwt
db.init_app(app) # 数据库

# 蓝图注册
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(api_bp, url_prefix='/api')

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return jsonify({'msg': 'Flask Backend Running'})

if __name__ == '__main__':
    app.run(debug=True)
