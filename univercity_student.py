import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt


mac2id = dict()  # 定义一个空字典，存储地址与对应顺序
onlinetimes = []  # 定义上网时间空列表
f = open('TestData.txt', encoding='utf-8')  # 打开txt文件，编码格式为utf-8
for line in f:
    messages = line.split(',')  # 信息分词
    mac = messages[2]   # 存储mac地址
    onlinetime = int(messages[6])  # 存储上网时间
    starttime = int(messages[4].split(' ')[1].split(':')[0])  # 开始上网的时间
    if mac not in mac2id:  # 新地址，新建
        mac2id[mac] = len(onlinetimes)
        onlinetimes.append((starttime, onlinetime))
    else:  # 旧地址，更新
        onlinetimes[mac2id[mac]] = [(starttime, onlinetime)]

real_x = np.array(onlinetimes).reshape(-1, 2)  # 修改成二维数据

# 按上网时间分类
X = real_x[:, 0:1]
# print(real_x[:, 0:1], real_x[:, 0])
# 两者的区别，0:1保证还是二维数据，还是列，0，将该列数据排成一行
db = skc.DBSCAN(eps=0.01, min_samples=20).fit(X)  # 邻域的大小，最小样本数生成db分类器
labels = db.labels_
print('Labels:', labels)
raito = len(labels[labels[:] == -1])/len(labels)  # 计算噪声点的比例
print('Noise raito :', format(raito, '.2%'))
n_clusters_ = len(set(labels))-(1 if -1 in labels else 0)  # 集合计算除噪音外还剩下几类
print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
for i in range(n_clusters_):
    print('Cluster ', i, ':')
    print(list(X[labels == i].flatten()))
plt.hist(X, 24)
plt.show()
# 按上网时长分类
Y = np.log(1+real_x[:, 1:])
db1 = skc.DBSCAN(eps=0.14, min_samples=10).fit(Y)
labels1 = db1.labels_
print('Labels:', labels1)
raito1 = len(labels1[labels1[:] == -1])/len(labels1)  # 计算噪声点的比例
print('Noise raito :', format(raito1, '.2%'))
n_clusters_1 = len(set(labels1))-(1 if -1 in labels1 else 0)  # 集合计算除噪音外还剩下几类
print('Estimated number of clusters: %d' % n_clusters_1)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Y, labels1))
for i in range(n_clusters_1):
    print('Cluster ', i, ':')
    print('\t number of sample: ', len(Y[labels1 == i]))
    print('\t mean of sample :', format(np.mean(real_x[labels1 == i][:, 1]), '.1f'))
    print('\t std of sample :', format(np.std(real_x[labels1 == i][:, 1]), '.1f'))
plt.hist(Y, 50)

plt.show()

#不错的技巧
# raito = len(labels[labels[:] == -1])/len(labels)
# n_clusters_ = len(set(labels))-(1 if -1 in labels else 0)
# ret_data.append([float(items[i]) for i in range(1, len(items))])