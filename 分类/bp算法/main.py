import numpy as np


class Bp:

    def sigmoid(self,x,w):

        return 1/(1+np.exp(-w.dot(x)))

    def loss(self, x, w, y):

        return ((self.sigmoid(x, w)-y).T.dot(self.sigmoid(x, w)-y))/2

    def grad(self, x, w, y):

        outer = self.sigmoid(x, w)*(1-self.sigmoid(x, w)).dot(x.T)

        inner = w.dot(x).dot(x.T)-y.dot(x.T)

        return inner*outer


if __name__ == '__main__':

    w = np.array([[0.4, 0.4, 0.6],
                  [0.5, 0.5, 0.6]])  # 2*3
    x = np.array([[0.574],
                 [0.574],  # 3*1
                  [1]])
    y = np.array([[0], [1]])

    bp = Bp()

    init_loss = bp.loss(x, w, y)
    print(init_loss)
    # while True:
    #     w -= 0.5*bp.grad(x, w, y)
    #     new_loss = bp.loss(x, w, y)
    #     if np.allclose(new_loss, init_loss):
    #         break
    #     init_loss = new_loss
    ws=w-0.5*bp.grad(x,w,y)
    print(ws)
    print(bp.sigmoid(x, w))

    
