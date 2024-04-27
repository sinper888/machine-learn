import numpy as np 
# from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def softmax(x,w):
    '''
    softmax函数具有归一化的功能 所以不用担心exp函数 溢出的情况
    '''
    fs=np.exp(x.dot(w))
    return np.divide(fs,np.sum(fs,axis=1,keepdims=True))

def loss(x,w):

    return (-1/m)*(mask*np.log(softmax(x,w))).sum()

def grad(x,w):

    return (1/m)*x.T.dot(softmax(x,w)-mask)

if __name__ == '__main__':
    
    # com=load_iris()
    com=load_digits()

    data,label=com['data'],com['target']

    x,tesx,y,tesy=train_test_split(data,label,train_size=0.7,
                                   shuffle=True)
    
    m,n=x.shape
    k=len(np.unique(y))
    w=np.zeros((n,k))

    mask=np.eye(k)[y]

    init_loss=loss(x,w)

    lr=0.0001 # 不同的数据集 可能需要修改学习率 才能更好的效果···

    while True:
        w-=lr*grad(x,w)

        new_loss=loss(x,w)

        if np.allclose(init_loss,new_loss):
            break
        init_loss=new_loss

    yp=np.argmax(softmax(x,w),axis=1).ravel()

    acc=(yp==y).sum()/len(y)

    print(acc)

    ty=np.argmax(softmax(tesx,w),axis=1).ravel()

    ac=(ty==tesy).sum()/len(tesy)

    print(ac)


              


