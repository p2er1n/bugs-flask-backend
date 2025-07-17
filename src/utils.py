import requests

def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        binary_data = response.content
        return binary_data
    else:
        raise ValueError("下载文件失败：",response.status_code)
    