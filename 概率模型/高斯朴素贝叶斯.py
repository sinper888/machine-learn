from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import numpy as np 
import pandas as pd

'''
计算所有样本依据类别分组 分别计算 每个特征的均值和方差 看成分分类模型
将新的测试数据 放入到模型 就可以计算出该样本在分类模型的概率分布 
在将概率分布加上先验概率 最组成最重的概率 取概率最大值的类别
'''

def fun1(df,ser):

    id=df.name
    men=df['mean'].values
    var=df['var'].values+EPS
    ser=ser.values

    '''
    每个测试样本 在高斯模型下的对数概率分布(转对数方便计算)
    '''
    t=np.log((1/np.sqrt(2*np.pi*var))*np.exp(-(ser-men)**2/(2*var))).sum()

    yp=yps[id]
    return t+yp


def test(ser):
    '''
    将测试数据(一条) 放入到 组合好的均值方差模型里 计算出测试数据在模型的概率分布
    '''
    p=groups.apply(fun1,ser)
    return p.idxmax()


if __name__ == '__main__':
    
    com=load_iris()
    data,label=com['data'],com['target']
    EPS=1e-8
    
    x,tesx,y,tesy=train_test_split(data,label,train_size=0.7,
                                    shuffle=True)
    m,n=x.shape
    colname=[f'x{i}' for i in range(n)]
    df=pd.DataFrame(x,columns=colname)
    y=pd.Series(y)
    yps=np.log(y.value_counts()/len(y)) # 先验概率
    
    means=df.groupby(y).apply(lambda x:x.mean(axis=0)) #按类别统计每个特征的均值
    vars=df.groupby(y).apply(lambda x:x.var(axis=0,ddof=0))#按类别统计每个特征的方差
    
    var_mean=pd.concat([vars,means],axis=1,keys=['var','mean'])
    '''
    将相同类别的均值和方差组合到一起
    '''
    
    groups=var_mean.groupby(level=0,as_index=False)

    tesx=pd.DataFrame(tesx,columns=colname)

    res=tesx.apply(test,axis=1).values.ravel()

    acc=(res==tesy).sum()/len(tesy)
    
    print(f'测试集准确率  {acc}')




