import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as man 

'''
梯度下降 一般用来求极值  但是 当f(x) 存在<=0的点时 
令： F(x)=f(x)**2 
   则 F(x)>=0

   此时F(x)一阶导数==0则是极值点 同时该点也是f(x)的根
'''

f=lambda x: 3*x**2+np.exp(x)/5+x**3-20

F=lambda x:f(x)**2

def grad(x):

    delta=0.0001
    return (F(x+delta)-F(x))/delta

def paint(n):

    x=x_picks[n]
    y=y_picks[n]
    p.set_data([x],[y])


if __name__ == '__main__':
    
    x=0
    init=F(x)
    alpha=0.0001
    xlist=[x]

    while True:
        x=x-alpha*grad(x)
        xlist.append(x)
        new=F(x)
        if np.allclose(new,init):
            print(x)
            break
        init=new

    m=len(xlist)
    x_picks=np.array(xlist)[np.linspace(0,m-1,10).astype(np.int32)]
    y_picks=f(x_picks)

    px=np.linspace(-5,5,10)
    py=f(px)

    fig=plt.figure()

    plt.plot(px,py)

    p=plt.plot([],[],'r*',markersize=10)[0]

    fs=man.FuncAnimation(fig,paint,np.arange(len(x_picks)),interval=1000)
    plt.show()

    












