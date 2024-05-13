import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import deque

class LogisticRegression:

    def __init__(self,x):

        self.x=x
        self.m=x.shape[0]
        
    @staticmethod
    def make_mask(x,y):
        
        m,n=x.shape
        k=len(np.unique(y))
        mask=np.eye(k)[y]
        w=np.zeros((n,k))

        return w,mask
    
    def softmax(self,w):

        fs=np.exp(self.x.dot(w))

        return np.divide(fs,np.sum(fs,axis=1,keepdims=True))
    
    def loss(self,w,mask):

        return (-1/self.m)*(np.log(self.softmax(w))*mask).sum()
    
    def grad(self,w,mask):

        return (1/self.m)*(self.x.T.dot(self.softmax(w)-mask))
    
    def predict(self,x,w):

        fs=np.exp(x.dot(w))

        return np.divide(fs,np.sum(fs,axis=1,keepdims=True))
    

def normal(w):

    dw=logist.grad(w,mask)
    w=w-lr*dw
    return w


def moment(w,theta):

    beta=0.9
    dw=logist.grad(w,mask)
    theta=beta*theta+lr*dw
    w=w-theta
    return w,theta

def new_moment(w,theta):

    beta=0.9
    dw=logist.grad(w+beta*theta,mask)
    theta=beta*theta+lr*dw
    w=w-theta
    return w,theta


def adagrad(w,r):

    rho=0.9
    dw=logist.grad(w,mask)

    r=rho*r+(1-rho)*np.power(dw,2)

    w=w-lr/(np.sqrt(r)+1e-4)*dw

    return w,r

def rmsp(w,theta,r,s):

    rho=0.9
    beta=0.9

    dw=logist.grad(w,mask)

    s=rho*s+(1-rho)*dw
    r=rho*r+(1-rho)*np.power(dw,2)
    theta=beta*theta+lr/(np.sqrt(r-s**2+1e-4))*dw
    w=w-theta

    return w,theta,r,s

def adam(w,s,r,i):

    beta1=0.9
    beta2=0.9
    dw=logist.grad(w,mask)

    s=beta1*s+(1-beta1)*dw
    r=beta2*r+(1-beta2)*np.power(dw,2)
    st=s/(1-beta1**i)
    rt=r/(1-beta2**i)
    w=w-(lr*st)/(np.sqrt(rt)+1e-4)

    return w,r,s


def amsgrad(w,s,r,theta,i):

    beta1=0.9
    beta2=0.9

    dw=logist.grad(w,mask)
    
    s=beta1*s+(1-beta1)*dw
    r=beta2*r+(1-beta2)*np.power(dw,2)

    theta=np.maximum(r,theta)

    st=s/(1-beta1**i)
    rt=theta/(1-beta2**i)

    w=w-lr*st/(np.sqrt(rt)+1e-4)

    return w,s,r,theta


    
if __name__ == '__main__':
    
    com=load_iris()
    data,label=com['data'],com['target']

    x,tesx,y,tesy=train_test_split(data,label,train_size=0.7,
                                   shuffle=True,)

    logist=LogisticRegression(x)

    w,mask=logist.make_mask(x,y)

    losses=deque(maxlen=5)

    init_loss=logist.loss(w,mask)

    losses.append(init_loss)

    lr=0.01

    theta,r,s,i=0,0,0,1

    while True:

        print(i)
        # w=normal(w)
        # w,theta=moment(w,theta)
        # w,theta=new_moment(w,theta)

        # w,r=adagrad(w,r)  # 当设置adagrad/rmsp算法时 放松allclose为 1e-3

        # w,theta,r,s=rmsp(w,theta,r,s)

        w,s,r=adam(w,r,s,i)

        # w,s,r,theta=amsgrad(w,s,r,theta,i)

        new_loss=logist.loss(w,mask)

        losses.append(new_loss)

        if np.allclose(losses[0],losses,1e-3):
            break
        i+=1

    yp=np.argmax(logist.predict(tesx,w),axis=1)

    acc=accuracy_score(yp,tesy)
    print(acc)

    

