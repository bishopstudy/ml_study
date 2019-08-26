import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

data = load_iris()  # 以字典形式加载鸢尾花数据集

y = data.target  # y表示数据集的标签
x = data.data  # x 保存数据集的数据
pca = PCA(n_components=2)  # 加载PCA算法，设置降维后数据为2
reduce_X = pca.fit_transform(x)  # 降维，保存
print(y,x)
print(reduce_X)

red_x, red_y = [], []
green_x, green_y = [], []
blue_x, blue_y = [], []

# 数据存入
for i in range(len(reduce_X)):
    if y[i] == 0:
        red_x.append(reduce_X[i][0])
        red_y.append(reduce_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduce_X[i][0])
        blue_y.append(reduce_X[i][1])
    else:
        green_x.append(reduce_X[i][0])
        green_y.append(reduce_X[i][1])

# 绘制散点图
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')

# 展示
plt.show()

# plt.scatter(reduce_X[:,0], reduce_X[:, 1])
# plt.show()
