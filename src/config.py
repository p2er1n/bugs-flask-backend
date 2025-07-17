import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    # SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
    #     'sqlite:///' + os.path.join(basedir, '../instance/a.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    BASE_FILE_URL = os.getenv('BASE_FILE_URL')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'instance', 'uploads')
    AI_MODEL_API_URL = "http://47.104.137.214:19312/v1/chat/completions"
    AI_MODEL_NAME = "chatglm3"
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    

