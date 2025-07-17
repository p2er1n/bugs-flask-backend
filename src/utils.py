import requests
from flask import jsonify

def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        binary_data = response.content
        return binary_data
    else:
        raise ValueError("下载文件失败：",response.status_code)

def success(data=None, msg="success"):
    """
    生成一个成功的API响应。
    """
    response = {
        'code': 1,
        'msg': msg,
        'data': data if data is not None else {}
    }
    return jsonify(response)

def error(msg, code=0):
    """
    生成一个错误的API响应。
    """
    response = {
        'code': code,
        'msg': msg,
        'data': None
    }
    return jsonify(response)