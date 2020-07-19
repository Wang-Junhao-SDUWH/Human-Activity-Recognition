import json
import matplotlib.pyplot as plt
from tsfeature import feature_core
import csv


# 将json文件中的数据提取为列表
def extract_data(docs):
    data = []
    for i in range(len(docs)):
        f = open(docs[i], 'r')
        temp = []
        for line in f.readlines():
            d = json.loads(line)
            del d['_id']
            del d['_openid']
            del d['t_g']
            del d['t_a']
            d['class'] = i
            temp.append(d)
        data.append(temp)
    return data

def draw(seven, axis_name):
    index = [321, 322, 323, 324, 325, 326]
    for i in range(len(axis_name)):
        axis = seven[axis_name[i]]
        x = [j for j in range(len(axis))]
        plt.subplot(index[i])
        plt.plot(x, axis)
    plt.show()

# 将数据整形为 4（动作数）*6（轴数）*n（时间序列长度）的列表
def data_reshape(data, axis_name):
    d = []
    for i in range(len(data)):
        t = []
        for ax in axis_name:
            print(ax)
            tt = []
            for j in data[i]:
                # 数据来源为安卓端，每次采集持续10s，大约可以采到220个数据
                # 这一步将分离的10s的数据合到一起
                tt = tt + list(j[ax][-200:])
            t.append(tt)
        d.append(t)
    return d

#提取特征值
def extract_features(data):
    feas = []
    for i in range(len(data)):
        t = []
        for a in data[i]:
            t.append(feature_core.sequence_feature(a, 32, 16))
        temp = []
        for j in range(len(t[0])):
            b = []
            for k in range(len(t)):
                b += t[k][j]
            temp.append(b+[i])
        feas += temp
    return feas

#生成csv文件
def generate_csv(features):
    with open('./features.csv','a',newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in features:
            writer.writerow(row)


#docs中的是微信小程序数据库中直接导出的json文件，分别储存四个动作的时间序列信息
docs = ['./data/0.json', './data/1.json', './data/2.json', './data/3.json']
axis_name = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
data = extract_data(docs)
data = data_reshape(data, axis_name)
features = extract_features(data)
generate_csv(features)

