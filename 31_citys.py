import numpy as np
from sklearn.cluster import KMeans


# 自定义函数，读取txt文件
def load_data(filename):
    fr = open(filename, 'r+')  # 以“r+”打开文件
    # lines = fr.readlines()     # 按行读取，没必要
    ret_data = []              # 定义空列表，储存信息
    ret_city_name = []         # 定义空列表，储存城市名
    for line in fr:
        # items = line.strip().split(",")  # 按逗号分词
        items = line.split(",")  # 按逗号分词
        ret_city_name.append(items[0])   # 追加城市名
        ret_data.append([float(items[i]) for i in range(1, len(items))])  # 把数据追加上去

    return ret_data, ret_city_name


if __name__ == '__main__':
    data, cityName = load_data('city.txt')
    km = KMeans(n_clusters=3)  # 定义一个三个类中心的分类器
    label = km.fit_predict(data)  # 为不同大的数据分配标签（0,1,2）
    # print(label)
    # print(km.cluster_centers_)  # 中心数据
    expenses = np.sum(km.cluster_centers_, axis=1)  # 计算类簇中心的消费水平
    # print(expenses)
    city_cluster = [[], [], []]
    for j in range(len(cityName)):
        city_cluster[label[j]].append(cityName[j])  # 按类簇的序号将城市的名称分类
    for k in range(len(city_cluster)):              #打印信息
        print("Expense:%.2f" % expenses[k])
        print(city_cluster[k])
