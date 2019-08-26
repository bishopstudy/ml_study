import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState

n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)
dataset = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))
faces = dataset.data
print(faces.shape)
plt.imshow(faces[0].reshape(image_shape))
# a = dataset.images
# print(a.shape)
# for i in range(a.shape[0]):
#     plt.imshow(a[i]);
#     plt.show()


def plot_gallery(title, image, col=n_col, row=n_row):
    plt.figure(figsize=(2. * col, 2.26 * row))  # 创建图片，指定大小，英寸
    plt.suptitle(title, size=16)  # 设置标题字号大小

    for i, comp in enumerate(image):  # 运用enumerate自动编号
        plt.subplot(row, col, i+1)  # 绘制子图
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)  # 将data数据变成图片的格式，数值归一化，以灰度图显示
        plt.xticks(())  # 去除子图的坐标轴显示
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)  # 对子图位置及间隔进行调整


estimators = [
    ('Eigenfaces - PCA using SVD',
     decomposition.PCA(n_components=6, whiten=True)),
    ('Non-negative components - NMF',
     decomposition.NMF(n_components=6, init='nndsvda',
                       tol=5e-3))
]

for name, estimator in estimators:
    estimator.fit(faces)
    components_ = estimator.components_
    plot_gallery(name, components_[:n_components])

plt.show()
