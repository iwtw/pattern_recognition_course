import numpy as np
import matlabplot as plot

LEARNING_RATE = 1e-6
N1 = 440
N2 = 400
DIM = 2 + 1 

class Perceptron:
    def __init__( w ):
        self.w = np.random.randn( DIM )
    def loss ( w , x , y  ):
        "denoting N the number of samples , M the dimension of each sample"
        "w shapes [M]"
        "x shapes [M,N] , the augmented matrice"
        "y shapes [N] "
        M , N = x.shape()
        w = w.reshape( (M,1) )
        x = x.reshape( (M,N) )
        y = y.reshape( (N,1) )
        return y.dot( w.T).dot(x) 
    def generate_data(  ):
        x1 = np.ones( ( DIM , N1 ) )
        x2 = np.ones ( ( DIM , N2 ) )

        x1[0,:] = -1.7 + 1.1 * np.random.randn(N1)
        x1[1,:] = 1.6 + 0.9 * np.random.randn(N1)

        x2[0,:] = 1.3 + 1.0 * np.random.randn(N2)
        x2[1,:] = -1.5 + 0.8 * np.random.randn(N2)
        
        y1 = np.ones((N1))
        y2 = -1*np.ones((N2))
        return x1 , y1 ,  x2 , y2

if __name__ == "__main__" :
    x1 , y1 ,  x2 , y2  = generate_data()

