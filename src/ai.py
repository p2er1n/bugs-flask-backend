from flask import Blueprint, current_app
import time
import requests
import json

ai_bp = Blueprint('ai', __name__)

def stream_chat(messages):
    """
    连接到符合 OpenAI API 标准的 AI 模型，并流式返回响应。
    """
    api_url = current_app.config['AI_MODEL_API_URL']
    model_name = current_app.config['AI_MODEL_NAME']
    api_key = current_app.config['OPENAI_API_KEY']

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
        'Authorization': f'Bearer {api_key}'
    }
    
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.9,
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, stream=True, timeout=120)
        response.raise_for_status()  # 如果响应状态码不是 2xx，则抛出异常

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data:'):
                    try:
                        # 移除 'data: ' 前缀并解析 JSON
                        json_str = decoded_line[len('data: '):]
                        if json_str.strip() == '[DONE]':
                            break
                        data = json.loads(json_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0].get('delta', {}).get('content')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        # 忽略无法解析的行
                        print(f"Warning: Could not decode JSON from line: {decoded_line}")
                        continue
                        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to AI model: {e}")
        yield "抱歉，连接AI服务时出现错误。"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        yield "抱歉，发生了一个未知错误。" 