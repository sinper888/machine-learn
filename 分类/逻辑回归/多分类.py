from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np 

def softmax(x,w):

    f=x.dot(w)
    if (f>500).any():
        inx,iny=np.where(f>500)
        absf=-np.abs(f)
        fs=np.exp(absf)
        fs[inx,iny]=1/(fs[inx,iny]+EPS)
    else:
        fs=np.exp(f)
    
    return fs/fs.sum(axis=1,keepdims=True)

def loss(x,w):

    return (-1/m)*(mask*x.dot(w)+np.log(softmax(x,w)+EPS)).sum()

def grad(x,w):

    return (1/m)*(x.T.dot(softmax(x,w)-mask))

if __name__ == '__main__':
    
    EPS=1e-5
    com=load_digits()
    data,target=com['data'],com['target']

    x,tesx,y,tesy=train_test_split(data,target,train_size=0.7,
                                   shuffle=True)
    m,n=x.shape
    cls=len(np.unique(y))
    w=np.zeros((n,cls))
    mask=np.eye(cls)[y]

    init_loss=loss(x,w)
    lr=0.001
    while True:
        w-=lr*grad(x,w)
        new_loss=loss(x,w)
        if np.allclose(new_loss,init_loss):
            break
        init_loss=new_loss
    
    pred_train=np.argmax(x.dot(w),axis=1)
    train_acc=(pred_train==y).sum()/len(y)

    print(train_acc)

    pred_test=np.argmax(tesx.dot(w),axis=1)
    test_acc=(pred_test==tesy).sum()/len(tesy)

    print(test_acc)


