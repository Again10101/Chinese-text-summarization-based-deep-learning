from flask import Flask, request, jsonify
from flask_cors import CORS
from single import Single
from textrank_single import single
app = Flask(__name__)
CORS(app)  # 允许跨域

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    original_text = data.get('text', '')
    print(original_text)
    summary = Single(original_text)  # 调用你的摘要生成函数
    print(summary)
    return jsonify({'summary': summary})
@app.route('/summarize_alt', methods=['POST'])  # 新路由
def textrank_single():
    data = request.json
    original_text = data.get('text', '')
    print(original_text)
    summary = single(original_text)
    print(summary)
    return jsonify({'summary': summary})
if __name__ == '__main__':
    app.run()


