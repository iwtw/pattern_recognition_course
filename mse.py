import numpy as np
import matplotlib.pyplot as plot

class MSE:
    def __init__(self, X , y ):
        self.w = np.inv( X.T.dot(X) ).dot(X.T).dot(y)
    def g(self,X):
        return X.dot(self.w)
    
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

if __name__ == "__main__":
    X = np.concatenate( [x1,x2] , axis = 1 )
    X = X.T
    y = np.concatenate( [y1,y2] , axis = 0 )

    mse = MSE( X , y )

    temp = np.linspace(-6,6,100)
    X_ = np.ones( 100 ,3 )
    X_[:,0] = temp
    X_[:,1] = temp
    Z = mse.g(X_ )
    plot.contour(  temp  , temp , Z , levels=[0] )
    plot.show()
    


    
