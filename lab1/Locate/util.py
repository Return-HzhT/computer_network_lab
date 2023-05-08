import numpy as np
import matplotlib.pyplot as plt


def get_rssi(data, devices):
    datas = [
        data[data[:, 0] == devices[0], 3], data[data[:, 0] == devices[1], 3],
        data[data[:, 0] == devices[2], 3]
    ]
    rssi = [
        datas[0].astype(np.float32).mean(), datas[1].astype(np.float32).mean(),
        datas[2].astype(np.float32).mean()
    ]
    return rssi


def read_dists(data, devices):
    datas = [
        data[data[:, 0] == devices[0], 4], data[data[:, 0] == devices[1], 4],
        data[data[:, 0] == devices[2], 4]
    ]
    dists = [
        datas[0].astype(np.float32).mean(), datas[1].astype(np.float32).mean(),
        datas[2].astype(np.float32).mean()
    ]
    return dists


def get_distance(loc_dev, loc):
    f = lambda x, y: ((x[0] - y[0])**2 + (x[1] - y[1])**2)**.5
    distance = [f(loc, loc_dev[0]), f(loc, loc_dev[1]), f(loc, loc_dev[2])]
    return distance


def linear_regress(rssi, dist):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(np.array(rssi).reshape(-1, 1), np.log(dist))
    a = model.coef_
    b = model.intercept_
    return a, b


data1, data2, data3, data4, data5 = np.load(
    "./data/train/wifi_data_0.npy"), np.load(
        "./data/train/wifi_data_1.npy"), np.load(
            "./data/train/wifi_data_2.npy"), np.load(
                "./data/train/wifi_data_3.npy"), np.load(
                    "./data/train/wifi_data_4.npy")

loc_dev = [(4.3, 1.4), (0.0, 11.0), (16.0, 11.0)]
devices = ['00f40443', '00f3f161', '00f40444']
loc1, loc2, loc3, loc4, loc5 = (4, 6.5), (11, 5.5), (8, 8), (4, 11), (12, 11)
gt = [loc1, loc2, loc3, loc4, loc5]
rssis = get_rssi(data1, devices) + get_rssi(data2, devices) + get_rssi(
    data3, devices) + get_rssi(data4, devices) + get_rssi(data5, devices)
dists = get_distance(loc_dev, loc1) + get_distance(
    loc_dev, loc2) + get_distance(loc_dev, loc3) + get_distance(
        loc_dev, loc4) + get_distance(loc_dev, loc5)
A1, N1 = linear_regress(rssis[3 + 0:13 + 0:3], dists[3 + 0:13 + 0:3])
A2, N2 = linear_regress(rssis[3 + 1:13 + 1:3], dists[3 + 1:13 + 1:3])
A3, N3 = linear_regress(rssis[3 + 2:13 + 2:3], dists[3 + 2:13 + 2:3])


def cal_position(dists, loc_dev):
    import math
    point = [0, 0]
    e = 0.2
    cx, cy = cx1, cy1 = cx2, cy2 = 0, 0
    for i in range(3):
        for j in range(i + 1, 3):
            dist_p = math.sqrt((loc_dev[i][0] - loc_dev[j][0]) *
                               (loc_dev[i][0] - loc_dev[j][0]) +
                               (loc_dev[i][1] - loc_dev[j][1]) *
                               (loc_dev[i][1] - loc_dev[j][1]))
            if dists[i] + dists[j] >= dist_p:
                dr = dist_p / 2 + (dists[i] * dists[i] -
                                   dists[j] * dists[j]) / (2 * dist_p)
                ddr = math.sqrt(abs(dists[i] * dists[i] - dr * dr))
                cx = loc_dev[i][0] + (loc_dev[j][0] -
                                      loc_dev[i][0]) * dr / dist_p
                cy = loc_dev[i][1] + (loc_dev[j][1] -
                                      loc_dev[i][1]) * dr / dist_p
                cos = -(loc_dev[j][1] - loc_dev[i][1]) / dist_p
                sin = (loc_dev[j][0] - loc_dev[i][0]) / dist_p

                cx1 = cx + ddr * cos
                cx2 = cx - ddr * cos
                cy1 = cy + ddr * sin
                cy2 = cy - ddr * sin

                k = 3 - i - j
                dev1 = math.sqrt((cx1 - loc_dev[k][0]) *
                                 (cx1 - loc_dev[k][0]) +
                                 (cy1 - loc_dev[k][1]) * (cy1 - loc_dev[k][1]))
                if dev1 <= dists[k] + e and dev1 >= dists[k] - e:
                    point[0] = cx1 + (loc_dev[k][0] -
                                      cx1) * (1 / 2 - dists[k] / (2 * dev1))
                    point[1] = cy1 + (loc_dev[k][1] -
                                      cx1) * (1 / 2 - dists[k] / (2 * dev1))
                    return np.array(point).reshape(-1)
                dev2 = math.sqrt((cx2 - loc_dev[k][0]) *
                                 (cx2 - loc_dev[k][0]) +
                                 (cy2 - loc_dev[k][1]) * (cy2 - loc_dev[k][1]))
                if dev2 <= dists[k] + e and dev2 >= dists[k] - e:
                    point[0] = cx2 + (loc_dev[k][0] -
                                      cx2) * (1 / 2 - dists[k] / (2 * dev2))
                    point[1] = cy2 + (loc_dev[k][1] -
                                      cx2) * (1 / 2 - dists[k] / (2 * dev2))
                    return np.array(point).reshape(-1)
            else:
                cx = loc_dev[i][0] + (loc_dev[j][0] - loc_dev[i][0]
                                      ) * dists[i] / (dists[i] + dists[j])
                cy = loc_dev[i][1] + (loc_dev[j][1] - loc_dev[i][1]
                                      ) * dists[i] / (dists[i] + dists[j])

            point[0] += cx
            point[1] += cy

    return (np.array(point) / 3).reshape(-1)


def calculate_distance(rssis):
    return np.array([
        np.exp(rssis[0] * A1 + N1),
        np.exp(rssis[1] * A2 + N2),
        np.exp(rssis[2] * A3 + N3)
    ])


def location(file):
    data = np.load(file)
    dists1 = calculate_distance(get_rssi(data, devices))
    dists2 = read_dists(data, devices)
    return cal_position(dists1, loc_dev), cal_position(dists2,
                                                       loc_dev), data[-1][2]


output = {}
GT = np.array([[12, 6], [8, 6.0], [5, 5.0], [5, 8], [5, 11], [8, 11], [11, 11],
               [11, 8]])
import json

tmp1 = tmp2 = tmp3 = np.array([8, 8.0])
import matplotlib.pyplot as plt

plt.figure()
locd = np.array(loc_dev)
plt.xlim(-2, 18)
plt.ylim(-2, 18)
plt.xticks(np.arange(-2, 18, 2))
plt.yticks(np.arange(-2, 18, 2))
plt.grid()
plt.scatter(locd[:, 0], locd[:, 1])
plt.plot([0, 0], [0, 16], color='black')
plt.plot([0, 16], [16, 16], color='black')
plt.plot([16, 16], [16, 0], color='black')
plt.plot([16, 0], [0, 0], color='black')
for i in range(8):
    l1, l2, t = location(f"./data/test/wifi_data_{i}.npy")
    output[t] = l1.tolist(), l2.tolist(), GT[i].tolist()
    if i != 0:
        plt.plot([tmp1[1], l1[1]], [tmp1[0], l1[0]], color='b')
        #plt.plot([GT[i][0], l1[0]], [GT[i][1], l1[1]], color='g')
        plt.plot([tmp2[1], l2[1]], [tmp2[0], l2[0]], color='g')
        #plt.plot([GT[i][0], l2[0]], [GT[i][1], l2[1]], color='g')
        plt.plot([tmp3[1], GT[i][1]], [tmp3[0], GT[i][0]], color='r')
    tmp1 = l1
    tmp2 = l2
    tmp3 = GT[i]

import json
with open("output.json", "w") as f:
    f.write(json.dumps(output))
plt.show()
