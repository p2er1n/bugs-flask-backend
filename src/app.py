import re
import base64
import hashlib
import os
from flask import Flask, jsonify, current_app
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO
from config import Config
from models import db
from auth import auth_bp
from routes import api_bp
from ai import ai_bp, stream_chat

app = Flask(__name__)
app.config.from_object(Config)

# 初始化 SocketIO
# socketio = SocketIO(app, cors_allowed_origins="*")
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",  # 在生产环境中建议指定具体来源，而不是"*"
    max_http_buffer_size=16 * 1024 * 1024  # 16 MB
)

# flask扩展初始化
jwt = JWTManager(app) # jwt
db.init_app(app) # 数据库

# 蓝图注册
app.register_blueprint(auth_bp, url_prefix='/user')
app.register_blueprint(api_bp, url_prefix='/file')
app.register_blueprint(ai_bp, url_prefix='/ai')

with app.app_context():
    db.create_all()

def save_base64_image(data_uri: str) -> str:
    """
    解码 base64 数据URI, 保存图片, 并返回文件在服务器上的路径。
    e.g., data_uri: "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    """
    try:
        header, encoded = data_uri.split(',', 1)
        file_ext = header.split(';')[0].split('/')[1]
        image_data = base64.b64decode(encoded)
        
        file_hash = hashlib.md5(image_data).hexdigest()
        filename = f"{file_hash}.{file_ext}"
        
        upload_folder = current_app.config['UPLOAD_FOLDER']
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                f.write(image_data)
        
        return file_path
    except Exception as e:
        current_app.logger.error(f"Error saving base64 image: {e}")
        return None

def sanitize_input(text: str) -> str:
    """
    移除字符串中不符合规则的字符。
    允许的字符包括：中英文字母、数字、下划线、空格和一些常见标点。
    """
    # 正则表达式，匹配所有不在白名单内的字符
    pattern = re.compile(r'[^\w\s\u4e00-\u9fa5.,?!;:()\'"\[\]{}@#$%^&*\-=_+]+')
    sanitized_text = re.sub(pattern, '', text)
    return sanitized_text

@socketio.on('chat')
def handle_chat_event(json_data):
    messages = json_data.get('messages')
    if not messages:
        return

    # OpenAI API 要求最新的消息在最后
    last_message = messages[-1]
    if last_message.get('role') == 'user':
        content = last_message.get('content')
        
        if isinstance(content, list):
            # 处理多模态内容
            for part in content:
                if part.get('type') == 'text':
                    sanitized_text = sanitize_input(part.get('text', ''))
                    part['text'] = f"{sanitized_text}/nothink"
                elif part.get('type') == 'image_url':
                    data_uri = part.get('image_url', {}).get('url')
                    if data_uri:
                        save_base64_image(data_uri) # 保存图片但保持原始data URI给模型
        elif isinstance(content, str):
            # 向后兼容纯文本内容
            sanitized_content = sanitize_input(content)
            last_message['content'] = f"{sanitized_content}/nothink"

    for char in stream_chat(messages):
        socketio.emit('response', {'data': char})
        socketio.sleep(0)

@app.route('/')
def index():
    return jsonify({'msg': 'Flask Backend Running'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')
