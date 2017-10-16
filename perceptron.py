import numpy as np
import matlabplot as plot

LEARNING_RATE = 1e-6
N1 = 440
N2 = 400
DIM = 2 + 1 

class Perceptron:
    def __init__( self ):
        self.w = np.random.randn( DIM , 1  )
    def loss (  self,x , y  ):
        "w shapes [DIM,1]"
        "x shapes [DIM,N] , the augmented matrice of training samples "
        "y shapes [N,1] , the labels"
        M , N = x.shape()
        x = x.reshape( (DIM,N) )
        y = y.reshape( (N,1) )
        return y.dot( self.w.T).dot(x) 
    def grad (self, x , y):
        return x.dot(y)
    def upadte ( self,x , y  ):
        self.w += -1.0 * LEARNING_RATE * grad(x,y)
        

def generate_data(  ):
    x1 = np.ones( ( DIM , N1 ) )
    x2 = np.ones ( ( DIM , N2 ) )

    x1[0,:] = -1.7 + 1.1 * np.random.randn(N1)
    x1[1,:] = 1.6 + 0.9 * np.random.randn(N1)

    x2[0,:] = 1.3 + 1.0 * np.random.randn(N2)
    x2[1,:] = -1.5 + 0.8 * np.random.randn(N2)
    
    y1 = np.ones((N1))
    y2 = -1 * np.ones((N2))
    return x1 , y1 ,  x2 , y2

if __name__ == "__main__" :
    x1 , y1 ,  x2 , y2  = generate_data()
    perceptron = Perceptron()
    for it in range(1000):
        perceptron.update()


