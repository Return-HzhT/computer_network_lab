import sqlite3

# 连接到SQLite数据库
# 数据库文件是wifi_data.db，如果文件不存在，会自动在当前目录创建
conn = sqlite3.connect("wifi_data.db")
cursor = conn.cursor()
# 创建wifi_data_list表
cursor.execute(
    "create table wifi_data_list(id varchar(256), mmac varchar(256), mac varchar(256), time varchar(256), rssi varchar(256), range varchar(256), primary key(id,mac,time))"
)

cursor.close()
conn.commit()
conn.close()