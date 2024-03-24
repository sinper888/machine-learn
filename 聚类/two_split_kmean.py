import numpy as np
from numpy.linalg import norm
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.animation as mati
from matplotlib import cm


def clst(data, k):

    clusts = []
    clusts.append(data[np.random.randint(0, len(data))])

    for i in range(1, k):
        dis = np.min(norm(data[:, None] - clusts, axis=2), axis=1)
        prob = dis / dis.sum()

        cum = np.cumsum(prob)
        r = np.random.rand()
        for i, p in enumerate(cum):
            if r < p:
                clusts.append(data[i])
                break

    return np.array(clusts)


def km(data, k=2):

    m, n = data.shape
    clusts = clst(data, k)
    tags = np.zeros((m, 2))

    flag = True
    while flag:
        flag = False

        distance = norm(data[:, None] - clusts, axis=2)
        dis = np.min(distance, axis=1)
        tag = np.argmin(distance, axis=1)

        if not (tags[:, 0] == tag).all():
            flag = True

        tags[:, 0] = tag
        tags[:, 1] = dis
        for i in range(len(clusts)):
            cls = data[np.nonzero(tags[:, 0] == i)[0]]
            clusts[i] = cls.mean(axis=0)

    return clusts, tags


def two_km(data, k):

    m, n = data.shape
    clusts = []
    cs, ts = [], []
    clusts.append(data.mean(axis=0))
    cs.append(np.array(clusts))
    tags = np.zeros((m, 2))
    ts.append(tags[:, 0].copy())

    while len(clusts) < k:

        SSE = np.inf
        for i in range(len(clusts)):
            cur_data = data[tags[:, 0] == i]
            new_clust, new_tag = km(cur_data)
            new_sse = new_tag[:, 1].sum()
            old_sse = tags[tags[:, 0] != i, 1].sum()
            _sse = old_sse + new_sse

            if _sse < SSE:
                sign = i
                mid_clust = new_clust
                mid_tag = new_tag
                SSE = _sse

        mid_tag[mid_tag[:, 0] == 1, 0] = len(clusts)
        mid_tag[mid_tag[:, 0] == 0, 0] = sign
        tags[tags[:, 0] == sign, :] = mid_tag

        ts.append(tags[:, 0].copy())

        clusts[sign] = mid_clust[0]
        clusts.append(mid_clust[1])
        cs.append(np.array(clusts.copy()))

    return cs, ts


def paint(n):

    center = cs[n]
    tag = ts[n]
    m = len(np.unique(tag))

    init.set_data([], [])

    for i in range(m):
        dat = x[tag == i]
        p[i].set_data(dat[:, 0], dat[:, 1])

    p_center.set_data(center[:, 0], center[:, 1])


if __name__ == '__main__':

    k = 5
    x, y = make_blobs(200, 2, centers=k, cluster_std=0.5)

    cs, ts = two_km(x, k)

    fig = plt.figure()

    init = plt.plot(x[:, 0], x[:, 1], 'o')[0]

    colors = cm.viridis(np.linspace(0, 1, k))

    p = []
    for i in range(k):
        p.append(plt.plot([], [], 'o', color=colors[i])[0])

    p_center = plt.plot([], [], 'r*', markersize=20)[0]

    anti = mati.FuncAnimation(fig,
                              paint,
                              frames=np.arange(len(cs)),
                              interval=2000,
                              repeat=False)

    plt.show()