import numpy as  np

fx=lambda x:3*x**2+np.exp(x)/5+x**3-20

def grad(x):
    delta=1e-6
    return (fx(x+delta)-fx(x))/delta

x=0
while True:
    x=x-fx(x)/grad(x)
    if np.allclose(fx(x),0):
        break
print(x)


