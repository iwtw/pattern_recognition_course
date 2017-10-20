import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from utils import *

LEARNING_RATE = 1e-6
N1 = 440
N2 = 400
DIM = 2 + 1 


class Perceptron:
    def __init__( self , learning_rate = 1e-6 , w = np.zeros(DIM) ):

        self.w = w
        self.learning_rate = learning_rate
    def loss (  self,x , y  ):
        
        "w shapes [DIM]"
        " x shapes [N,DIM] , the augmented matrice of training samples "
        "y shapes [N] , the labels"
        return  x.dot(self.w).dot(y)
    def grad (self, x , y):
        return y.dot(x)
    def update ( self,x , y  ):
        self.w += -1.0 * self.learning_rate * self.grad(x,y)
    def g(self,x):
        return  x.dot(self.w)


if __name__ == "__main__" :
    x1 , y1 ,  x2 , y2  = generate_data(N1 , N2 , DIM)

    X = np.concatenate( [x1,x2] ,axis = 0 )
    y = np.concatenate( [y1,y2], axis = 0 )


    X_ =np.ones((100*100,3))
    temp = np.linspace(-6,6,100)
    X_[:,0] = np.concatenate( [temp for i in range(100) ]  )
    X_[:,1] = [ i for i in temp  for j in range(100)]
    #Z = perceptron.g(X_)
    #Z = Z.reshape((100,100))

    learning_rate = [10, 1e-6 , 1e-20]
    for j in range(2):
        plot.figure()
        plot_data(x1,y1,x2,y2)
        for i in range(3):
            if j == 0 :
                w = np.random.randn(3)
                name = "lr:1e-06,w:r"
            else :
                w = 1e-3*np.ones(3)
                name = "lr:%.0e,w:1e-3"%(learning_rate[i])

            p = Perceptron( learning_rate[i] , w  )
            for it in range(100):
                p.update( X , y )
            
            Z = p.g(X_)
            Z_ = Z.reshape((100,100))
            plot_plane(temp ,  Z_ , name , color = np.random.uniform( 0,0.7,3 ) )
        plot.savefig("perceptron{}.png".format(j),dpi=200)
    #plot.show()

