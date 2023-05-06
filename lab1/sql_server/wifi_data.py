from flask import Flask, request, jsonify
import json
import sqlite3

app = Flask(__name__)
app.debug = True
conn = sqlite3.connect("wifi_data.db", check_same_thread=False)
cursor = conn.cursor()


@app.route('/wifi_post/', methods=['post'])  # 用来采集数据的url
def add_test():
    if not request.content_length:  #检测是否有数据
        return ('fail')
    str = request.form.get("data")
    data_json = json.loads(str)
    my_mac = '92:ec:f2:fa:7d:17'
    id = data_json['id']
    time = data_json['time']
    mmac = data_json['mmac']

    test_rssi = 0
    test_range = 0
    data_list = data_json['data']
    for i in data_list:
        if i['mac'] == my_mac:
            test_rssi = i['rssi']
            test_range = i['range']
            # 写入数据库
            value = '(\"' + id + '\",\"' + mmac + '\",\"' + my_mac + '\",\"' + time + '\",\"' + test_rssi + '\",\"' + test_range + '\")'
            sql_str = "insert into wifi_data_list(id, mmac, mac, time, rssi, range) values " + value
            cursor.execute(sql_str)
    print(id, mmac, test_rssi, test_range)
    return str


@app.route('/')  # 查看IP地址
def home():
    url = request.url_root
    print(url)
    return url


if __name__ == '__main__':
    app.run('0.0.0.0', 5000)  # 使用本地IP地址和5000端口

cursor.close()
conn.commit()
conn.close()
