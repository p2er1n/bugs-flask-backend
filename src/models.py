from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import create_access_token
from datetime import timedelta

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    openid = db.Column(db.String(120), unique=True, nullable=True)
    username = db.Column(db.String(80), unique=True, nullable=True)
    password = db.Column(db.String(120), nullable=True)
    avatar_url = db.Column(db.String(80), nullable=True)

    # 建立与 Picture 表的关系
    # picture = db.relationship('Picture', primaryjoin='User.avatar_id == foreign(Picture.id)', uselist=False)

    def get_token(self, expires_in=3600000):
        return create_access_token(identity=str(self.id), expires_delta=timedelta(seconds=expires_in))
    
    @staticmethod
    def get_by_openid(openid):
        return User.query.filter_by(openid=openid).first()
    
    @staticmethod
    def get_by_username(username):
        return User.query.filter_by(username=username).first()
    
    def save(self):
        db.session.add(self)
        db.session.commit()

class Picture(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False)    # OSS 的图片访问地址
    hash = db.Column(db.String(128), nullable=False)    # 图片哈希值，用于唯一标识或去重
    user_id = db.Column(db.Integer, nullable=False)  # 外键，指向用户 id
    
    # 建立与 User 表的关系
    user = db.relationship('User', primaryjoin='Picture.user_id == foreign(User.id)', backref=db.backref('pictures', lazy=True))

    def save(self):
        db.session.add(self)
        db.session.commit()

    @staticmethod
    def get_by_hash(file_hash):
        return Picture.query.filter_by(hash=file_hash).first()

    @staticmethod
    def get_by_id(pic_id):
        return Picture.query.get(pic_id)

    def to_dict(self):
        return {
            'id': self.id,
            'url': self.url,
            'hash': self.hash,
            'user_id': self.user_id
        }

class DetectionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    picture_id = db.Column(db.Integer, db.ForeignKey('picture.id'), nullable=False)
    result = db.Column(db.Text, nullable=False)  # JSON format of List<BoxVo>
    model_name = db.Column(db.String(80), nullable=True)
    detection_time = db.Column(db.DateTime, nullable=False)

    user = db.relationship('User', backref=db.backref('detection_histories', lazy=True))
    picture = db.relationship('Picture', backref=db.backref('detection_histories', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'picture_id': self.picture_id,
            'result': self.result,
            'result_image_url': self.picture.url,
            'model_name': self.model_name,
            'detection_time': self.detection_time.isoformat()
        }


