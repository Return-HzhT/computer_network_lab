from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

coordinates = [(1, 2), (3, 4), (5, 6)]

@app.route('/coordinates', methods=['GET'])
def get_coordinates():
    return jsonify(coordinates)

@app.route('/')
def index():
    return render_template('coordiantes.html')
    return render_template('temp.html')

@app.route('/data')
def get_data():

    data = open('./data/data.json').read()

    # 使用 jsonify 函数将 Python 对象转换为 JSON 格式
    json_data = jsonify(data)

    # 设置响应头，指定内容类型为 JSON
    headers = {
        'Content-Type': 'application/json'
    }

    # 返回 JSON 数据给前端
    return json_data, 200, headers

if __name__ == '__main__':
    app.run()