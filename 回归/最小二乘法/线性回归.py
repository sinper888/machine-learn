import numpy as np 
from numpy.linalg import inv
import matplotlib.pyplot as plt 


'''
根据损失函数 损失函数 计算一阶导数的根 

def loss(x,w,y):

    return 1/2*((x.dot(w)-y).T.dot(x.dot(w)-y))

'''
if __name__ == '__main__':
    
    x=np.linspace(0,20)
    y=2*x+10+np.random.randn(50)*5

    xs=np.c_[x,[1]*len(x)]
    ys=y[:,None]

    print(xs.shape,ys.shape)

    w=inv(xs.T.dot(xs)).dot(xs.T).dot(ys)

    pred=xs.dot(w)

    plt.scatter(x,y)
    plt.plot(xs[:,0],pred[:,0],'r--')

    plt.show()
    print(w)

