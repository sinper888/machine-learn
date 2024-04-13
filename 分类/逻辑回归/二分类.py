import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def sigmoid(x,w):
    f=-x.dot(w)
    if (f>500).any():
        inx,iny=np.where(f>500)
        absf=-np.abs(f)
        fs=np.exp(absf)
        fs[inx,iny]=1/(fs[inx,iny]+EPS)
    else:
        fs=np.exp(f)

    return 1/(1+fs)

def loss(x,w,y):

    return (-1/m)*((y*(x.dot(w))+np.log(1-sigmoid(x,w)+EPS)).sum())

def grad(x,w):

    return (1/m)*(x.T.dot(sigmoid(x,w)-y))

if __name__ == '__main__':

    EPS=1e-5
    com=load_breast_cancer()
    data,label=com['data'],com['target']

    x,tesx,y,tesy=train_test_split(data,label,train_size=0.7,
                                   shuffle=True)
    
    y=y[:,None]
    m,n=tesx.shape
    w=np.zeros((n,1))

    init_loss=loss(x,w,y)
    lr=0.0001
    while True:
        w-=lr*grad(x,w)
        new_loss=loss(x,w,y)
        if np.allclose(new_loss,init_loss):
            break
        init_loss=new_loss
    
    train_pred=sigmoid(x,w).ravel()

    y_pred=np.where(train_pred>=0.5,1,0)

    train_acc=(y_pred==y.ravel()).sum()/len(y.ravel())

    print(train_acc)

    test_pred=sigmoid(tesx,w).ravel()

    test_y=np.where(test_pred>=0.5,1,0)

    test_acc=(test_y==tesy).sum()/len(tesy)

    print(test_acc)



    

