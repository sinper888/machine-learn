import numpy as np 
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split   


def grad(x,w,y):

    return x.T.dot(x.dot(w)-y)


def Loss(x,w,y):

    return (x.dot(w)-y).T.dot(x.dot(w)-y)/2


if __name__ == '__main__':

    n=5
    
    x,y=make_regression(200,n)

    train_x,test_x,train_y,test_y=train_test_split(x,y,
                                                   train_size=0.7)
    
    train_y=train_y[:,None]
    test_y=test_y[:,None]
    
    w=np.zeros((n,1))
    alpha=0.001
    init=Loss(train_x,w,train_y)

    while True:
        
        w=w-alpha*grad(train_x,w,train_y)
        new=Loss(train_x,w,train_y)
        if np.allclose(new,init):
            print(w)
            break
        init=new

    pred_train=train_x.dot(w)

    print(np.allclose(pred_train,train_y,1e-3))

    pred_test=test_x.dot(w)

    print(np.allclose(pred_test,test_y,1e-3))


        


