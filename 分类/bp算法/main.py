import numpy as np 

class Bp:

    def __init__(self,x,y):
        
        self.x=x
        self.y=y

    def forward(self,w):

        x_=self.x
        fys=[]
        for var in w:
            p=self.sigmoid(x_,var)
            fys.append(p)
            x_=p
        return fys


    def sigmoid(self,x,w):

        return 1/(1+np.exp(-x.dot(w)))
    
    def loss(self,w):

        fys=self.forward(w)
        p=fys[-1]
        return (((p-self.y).T.dot(p-self.y))/2)
    
    def dsigmoid(self,y):

        return y*(1-y)
    
    def backward(self,fys,w,alpha):
        
        err=fys[1]-self.y  
        g1=fys[0].dot(err*self.dsigmoid(fys[1]))

        p=err.dot(w[1].T)        
        g0=self.x.T.dot(self.dsigmoid(fys[0])*p)

        w[1]=w[1]-alpha*g1
        w[0]=w[0]-alpha*g0

        return w
    
if __name__ == '__main__':

    x0=np.array([[0 ,0 ,1],
                 [1, 0 ,1],
                 [0, 1, 1],
                 [1 ,1 ,1]])
    
    y=np.array([[0],[1],[1],[0]])
    
    w1=np.random.rand(3,4)*5
    w1[:,-1]=1
    w2=np.random.rand(4,1)*5
    w2[:,-1]=1
    ws=[w1,w2]

    bp=Bp(x0,y)

    alpha=0.01

    while True:
        fys=bp.forward(ws)
        new=bp.backward(fys,ws,alpha)
        new_loss=bp.loss(new).ravel()[0]
        if new_loss<1e-2:
            break
        ws=new
    res=bp.forward(ws)[-1]
    print(res)


    

