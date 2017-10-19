import numpy as np
import matplotlib.pyplot as plot

LEARNING_RATE = 1e-6
N1 = 440
N2 = 400
DIM = 2 + 1 


class Perceptron:
    def __init__( self ):
        self.w = np.random.randn( DIM  )
    def loss (  self,x , y  ):
        
        "w shapes [DIM]"
        " x shapes [N,DIM] , the augmented matrice of training samples "
        "y shapes [N] , the labels"
        return  x.dot(self.w).dot(y)
    def grad (self, x , y):
        return y.dot(x)
    def update ( self,x , y  ):
        self.w += -1.0 * LEARNING_RATE * self.grad(x,y)
    def g(self,x):
        return  x.dot(self.w)


def generate_data(  ):
    x1 = np.ones( ( N1 , DIM ) )
    x2 = np.ones ( ( N2 , DIM ) )

    x1[:,0] = -1.7 + 1.1 * np.random.randn(N1)
    x1[:,1] = 1.6 + 0.9 * np.random.randn(N1)

    x2[:,0] = 1.3 + 1.0 * np.random.randn(N2)
    x2[:,1] = -1.5 + 0.8 * np.random.randn(N2)
    
    y1 = np.ones(N1)
    y2 = -1 * np.ones(N2)
    return x1 , y1 ,  x2 , y2

if __name__ == "__main__" :
    x1 , y1 ,  x2 , y2  = generate_data()
    plot.axis( (-6,6,-6,6) )
    plot.scatter(x1[:,0],x1[:,1],marker=".")
    plot.scatter(x2[:,0],x2[:,1],marker=".")
    X = np.concatenate( [x1,x2] ,axis = 0 )
    y = np.concatenate( [y1,y2], axis = 0 )

    perceptron = Perceptron()
    for it in range(1000):
        perceptron.update( X , y )
        print(perceptron.loss(X,y))

    hyperplane = np.zeros((100,100))
    for i_idx,i_v in enumerate (np.linspace(-6,6,100)):
        for j_idx ,j_v in enumerate ( np.linspace(-6,6,100)):
            g = perceptron.g( np.array( [i_v,j_v,1] ) )
            hyperplane[i_idx,j_idx] = g 

    i = j = np.linspace(-6,6,100)
    plot.contour(i,j,hyperplane,levels=[0])
    #plot.contour(np.array( hyperplane).T,levels=[0])
    #plot.scatter( hyperplane[0] , hyperplane[1],marker="." )
    plot.show()
