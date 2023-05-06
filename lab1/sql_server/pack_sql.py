import sqlite3
import numpy as np

conn = sqlite3.connect("wifi_data.db")
cursor = conn.cursor()
cursor.execute("select * from wifi_data_list")
result = cursor.fetchall()
cursor.close()
conn.commit()
conn.close()

id_lst = []
mmac_lst = []
mac_lst = []
time_lst = []
rssi_lst = []
range_lst = []

for i in result:
    id_lst.append(i[0])
    mmac_lst.append(i[1])
    mac_lst.append(i[2])
    time_lst.append(i[3])
    rssi_lst.append(i[4])
    range_lst.append(i[5])

final_lst = [[], [], [], [], []]
n = len(id_lst)
for i in range(n):
    hour = int(time_lst[i][11:13])
    minute = int(time_lst[i][14:16])
    if hour == 18 and 36 <= minute <= 41:
        final_lst[0].append(
            [id_lst[i], mmac_lst[i], time_lst[i], rssi_lst[i], range_lst[i]])
    elif hour == 18 and 42 <= minute <= 47:
        final_lst[1].append(
            [id_lst[i], mmac_lst[i], time_lst[i], rssi_lst[i], range_lst[i]])
    elif hour == 18 and 48 <= minute <= 53:
        final_lst[2].append(
            [id_lst[i], mmac_lst[i], time_lst[i], rssi_lst[i], range_lst[i]])
    elif (hour == 18 and 55 <= minute) or (hour == 19 and minute == 0):
        final_lst[3].append(
            [id_lst[i], mmac_lst[i], time_lst[i], rssi_lst[i], range_lst[i]])
    elif hour == 19 and 1 <= minute <= 6:
        final_lst[4].append(
            [id_lst[i], mmac_lst[i], time_lst[i], rssi_lst[i], range_lst[i]])

np.save("./db_data_npy/wifi_data_0", np.array(final_lst[0]))
np.save("./db_data_npy/wifi_data_1", np.array(final_lst[1]))
np.save("./db_data_npy/wifi_data_2", np.array(final_lst[2]))
np.save("./db_data_npy/wifi_data_3", np.array(final_lst[3]))
np.save("./db_data_npy/wifi_data_4", np.array(final_lst[4]))

cnt = 0
for i in range(5):
    path = "./db_data_npy/wifi_data_" + str(i) + ".npy"
    a = np.load(path)
    print(a)
    cnt += a.shape[0]
print(cnt)