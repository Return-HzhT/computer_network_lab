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
loc1, loc2, loc3, loc4, loc5 = (4, 6.5), (12, 5.5), (8, 8), (4, 11), (12, 11)
gt = [loc1, loc2, loc3, loc4, loc5]
rssis = get_rssi(data1, devices) + get_rssi(data2, devices) + get_rssi(
    data3, devices) + get_rssi(data4, devices) + get_rssi(data5, devices)
dists = get_distance(loc_dev, loc1) + get_distance(
    loc_dev, loc2) + get_distance(loc_dev, loc3) + get_distance(
        loc_dev, loc4) + get_distance(loc_dev, loc5)
A1, N1 = linear_regress(rssis[3::3], dists[3::3])
A2, N2 = linear_regress(rssis[4::3], dists[4::3])
A3, N3 = linear_regress(rssis[5::3], dists[5::3])
print(A1, A2, A3, N1, N2, N3)


def cal_position(dists, loc_dev):
    import math
    point = [0, 0]
    points = []
    points3 = []

    assert (len(dists) == 3)
    assert (len(loc_dev) == 3 and len(loc_dev[0]) == 2)
    e = 0.2
    found = False
    tmpx, tmpy = 0, 0
    tmpx1, tmpy1 = 0, 0
    tmpx2, tmpy2 = 0, 0
    for i in range(3):
        assert (dists[i] >= 0)
        if found:
            break
        for j in range(i + 1, 3):
            p2p = math.sqrt((loc_dev[i][0] - loc_dev[j][0]) *
                            (loc_dev[i][0] - loc_dev[j][0]) +
                            (loc_dev[i][1] - loc_dev[j][1]) *
                            (loc_dev[i][1] - loc_dev[j][1]))
            if dists[i] + dists[j] >= p2p:
                dr = p2p / 2 + (dists[i] * dists[i] -
                                dists[j] * dists[j]) / (2 * p2p)
                ddr = math.sqrt(abs(dists[i] * dists[i] - dr * dr))
                tmpx = loc_dev[i][0] + (loc_dev[j][0] -
                                        loc_dev[i][0]) * dr / p2p
                tmpy = loc_dev[i][1] + (loc_dev[j][1] -
                                        loc_dev[i][1]) * dr / p2p
                cos = -(loc_dev[j][1] - loc_dev[i][1]) / p2p
                sin = (loc_dev[j][0] - loc_dev[i][0]) / p2p

                tmpx1 = tmpx + ddr * cos
                tmpx2 = tmpx - ddr * cos
                tmpy1 = tmpy + ddr * sin
                tmpy2 = tmpy - ddr * sin
                points.append([tmpx1, tmpy1])
                points.append([tmpx2, tmpy2])
            else:
                tmpx = loc_dev[i][0] + (loc_dev[j][0] - loc_dev[i][0]
                                        ) * dists[i] / (dists[i] + dists[j])
                tmpy = loc_dev[i][1] + (loc_dev[j][1] - loc_dev[i][1]
                                        ) * dists[i] / (dists[i] + dists[j])

            if dists[i] + dists[j] >= p2p:
                k = 3 - i - j
                dev1 = math.sqrt((tmpx1 - loc_dev[k][0]) *
                                 (tmpx1 - loc_dev[k][0]) +
                                 (tmpy1 - loc_dev[k][1]) *
                                 (tmpy1 - loc_dev[k][1]))
                if dev1 <= dists[k] + e and dev1 >= dists[k] - e:
                    point[0] = tmpx1 + (loc_dev[k][0] -
                                        tmpx1) * (1 / 2 - dists[k] /
                                                  (2 * dev1))
                    point[1] = tmpy1 + (loc_dev[k][1] -
                                        tmpx1) * (1 / 2 - dists[k] /
                                                  (2 * dev1))
                    found = True
                    break
                dev2 = math.sqrt((tmpx2 - loc_dev[k][0]) *
                                 (tmpx2 - loc_dev[k][0]) +
                                 (tmpy2 - loc_dev[k][1]) *
                                 (tmpy2 - loc_dev[k][1]))
                if dev2 <= dists[k] + e and dev2 >= dists[k] - e:
                    point[0] = tmpx2 + (loc_dev[k][0] -
                                        tmpx2) * (1 / 2 - dists[k] /
                                                  (2 * dev2))
                    point[1] = tmpy2 + (loc_dev[k][1] -
                                        tmpx2) * (1 / 2 - dists[k] /
                                                  (2 * dev2))
                    found = True
                    break

            point[0] += tmpx
            point[1] += tmpy
            points3.append([tmpx, tmpy])

    if not found:
        point[0] /= 3
        point[1] /= 3

    return np.array(point).reshape(-1)


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
    return cal_position(dists1, loc_dev), cal_position(dists2, loc_dev)

