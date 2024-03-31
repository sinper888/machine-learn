import numpy as np
from numpy.linalg import inv
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

''''

迭代公式 w1=w-dloss1/dloss2
1 牛顿法 求解方程根使用
2 在线性回归中 设置好Loss函数以后 极值出现在 Loss 一阶导的根
3 将一阶导看成原函数 计算出 Loss的2阶导 
4 通过迭代出合适的 w的值

'''
# def loss(x, w, y):

#     return 1/2*(x.dot(w)-y).T.dot(x.dot(w)-y)


def grad1(x, w, y):

    return x.T.dot(x.dot(w)-y)


def grad2(x):

    return x.T.dot(x)


if __name__ == '__main__':

    x, y = make_regression(50, 1, noise=15, random_state=0)

    y = y[:, None]

    x = np.c_[x.ravel(), [1]*len(x)]
    m, n = x.shape
    w = np.zeros((n, 1))

    while True:
        w -= inv(grad2(x)).dot(grad1(x, w, y))
        if np.allclose(grad1(x, w, y), 0):
            break

    ypred = x.dot(w)

    plt.scatter(x[:, 0], y.ravel())

    plt.plot(x[:, 0], ypred.ravel(), 'r--')

    plt.show()
