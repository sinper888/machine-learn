import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Perceptrons:

    def __init__(self,n,y):

        self.w1=np.random.randn(n,8)
        self.w2=np.random.randn(8,5)
        self.w3=np.random.randn(5,1)

        self.y=y[:,None]

    def sigmoid(self,x):

        return 1/(1+np.exp(-x))
    

    def dsigmoid(self,x):

        return x*(1-x)
    
    def loss(self,y):

        return (y-self.y).T.dot(y-self.y)/2

    def forward(self,x):

        self.out1=self.sigmoid(x.dot(self.w1))
        self.out2=self.sigmoid(self.out1.dot(self.w2))
        self.out3=self.sigmoid(self.out2.dot(self.w3))

    def grad(self,x,lr):

        self.forward(x)

        err3=(self.out3-self.y)*self.dsigmoid(self.out3)

        err2=err3.dot(self.w3.T)*self.dsigmoid(self.out2)

        err1=err2.dot(self.w2.T)*self.dsigmoid(self.out1)

        self.w3-=lr*self.out2.T.dot(err3)
        self.w2-=lr*self.out1.T.dot(err2)
        self.w1-=lr*x.T.dot(err1)

if __name__ == '__main__':

    com=load_breast_cancer()

    data,label=com['data'],com['target']

    scale=MinMaxScaler()
    data=scale.fit_transform(data)

    x,tesx,y,tesy=train_test_split(data,label,train_size=0.7,
                                   shuffle=True)

    n=x.shape[1]

    per=Perceptrons(n,y)

    while True:
        per.grad(x,0.001)

        new_loss=per.loss(per.out3)
        # print(new_loss)

        if new_loss<1e1:
            break
    
    trainy=per.out3.ravel()

    trainy=np.where(trainy>=0.5,1,0)

    print(f'train: {(trainy==y).sum()/len(y)}')

    per.forward(tesx)

    testy=per.out3.ravel()

    testy=np.where(testy>=0.5,1,0)

    print(f'test {(testy==tesy).sum()/len(tesy)}')







    