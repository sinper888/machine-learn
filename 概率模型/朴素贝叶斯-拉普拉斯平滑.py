import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer


def fun1(df):

    counts = df.apply(lambda x: x.value_counts())

    return counts.T


def train(x, y, alpha=1):  # 返回 条件概率模型 (不同分类标签下 不同x的分布下的各个概率)

    counts = df.groupby(y).apply(fun1)

    counts.fillna(0, inplace=True)

    uni_x = x.apply(lambda x: len(x.unique()*alpha), axis=0)

    ser = pd.Series(np.tile(uni_x, num_class), index=counts.index)

    res = (counts+alpha).div(counts.sum(axis=1)+ser, axis=0)

    return np.log(res)


def fun3(df, ser):
    '''
    计算每个样本的在所有类别的概率
    '''

    id = df.name

    dt = df.values
    ser = ser.values
    ds = dt[range(len(dt)), ser]  # 根据样本的分布 在条件概率找出对应的概率
    res = ds.sum()+pys[id]  # 因为所有的概率 已转成对数形式 所以由累乘变为累加 最后再加上先验概率
    return res


def test(ser):

    ser = ser.astype('int')
    p = groups.apply(fun3, ser)

    return p.idxmax()


if __name__ == '__main__':

    com = load_iris()
    data, label = com['data'], com['target']

    '''
    数据分箱操作,将连续变量转成离散形式的
    '''
    kb = KBinsDiscretizer(n_bins=3)
    data = kb.fit_transform(data).todense()

    x, tesx, y, tesy = train_test_split(
        data, label, train_size=0.7, shuffle=True)

    m, n = x.shape
    colname = [f'x{i}' for i in range(n)]

    df = pd.DataFrame(x, columns=colname)
    y = pd.Series(y)

    num_class = len(y.unique())

    alpha = 1  # 平滑系数 =1时  为拉普拉斯平滑
    model = train(df, y)

    groups = model.groupby(level=0, as_index=False)

    py = y.value_counts()  # 不同类别的先验概率
    pys = np.log((py+1)/(py.sum()+num_class))

    tesx = pd.DataFrame(tesx, columns=colname)
    y_pred = tesx.apply(test, axis=1).values.ravel()

    print((y_pred == tesy).sum()/len(tesy))
