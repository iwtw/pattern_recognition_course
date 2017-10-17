import numpy as np
import matplotlib.pyplot as plot

LEARNING_RATE = 1e-6
N1 = 440
N2 = 400
DIM = 2 + 1 


class Perceptron:
    def __init__( self ):
        self.w = np.random.randn( DIM , 1  )
    def loss (  self,x , y  ):
        "
        w shapes [DIM,1]
        x shapes [DIM,N] , the augmented matrice of training samples 
        y shapes [N,1] , the labels
        "
        _ , N = x.shape()
      #  x = x.reshape( (DIM,N) )
      #  y = y.reshape( (N,1) )
        return y.dot( self.w.T).dot(x) 
    def grad (self, x , y):
        return x.dot(y)
    def update ( self,x , y  ):
        self.w += -1.0 * LEARNING_RATE * self.grad(x,y)
    def g(self,x):
        return self.w.T.dot(x)


def generate_data(  ):
    x1 = np.ones( ( DIM , N1 ) )
    x2 = np.ones ( ( DIM , N2 ) )

    x1[0,:] = -1.7 + 1.1 * np.random.randn(N1)
    x1[1,:] = 1.6 + 0.9 * np.random.randn(N1)

    x2[0,:] = 1.3 + 1.0 * np.random.randn(N2)
    x2[1,:] = -1.5 + 0.8 * np.random.randn(N2)
    
    y1 = np.ones((N1,1))
    y2 = -1 * np.ones((N2,1))
    return x1 , y1 ,  x2 , y2

if __name__ == "__main__" :
    x1 , y1 ,  x2 , y2  = generate_data()
    plot.axis( (-6,6,-6,6) )
    plot.scatter(x1[0,:],x1[1,:],marker=".")
    plot.scatter(x2[0,:],x2[1,:],marker=".")
    perceptron = Perceptron()
    X = np.concatenate( [x1,x2] ,axis = 1 )
    Y = np.concatenate( [y1,y2], axis = 0 )
    for it in range(1000):
        perceptron.update( X , Y )
    hyperplane = np.zeros((100,100))
    for i_idx,i_v in enumerate (np.linspace(-6,6,100)):
        for j_idx ,j_v in enumerate ( np.linspace(-6,6,100)):
            g = perceptron.g( np.array( [i_v,j_v,1] ).T )
            hyperplane[i_idx,j_idx] = g 

    i = j = np.linspace(-6,6,100)
    plot.contour(i,j,hyperplane,levels=[0])
    #plot.contour(np.array( hyperplane).T,levels=[0])
    #plot.scatter( hyperplane[0] , hyperplane[1],marker="." )
    plot.show()
