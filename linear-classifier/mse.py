import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
from utils import *

N1 = 440
N2 = 400
DIM = 2 + 1 

class MSE:
    def __init__(self, X , y ):
        self.w = np.linalg.inv( X.T.dot(X) ).dot(X.T).dot(y)
    def g(self,X):
        return X.dot(self.w)
    
if __name__ == "__main__" :
    x1 , y1 ,  x2 , y2  = generate_data(N1 , N2 , DIM)

    X = np.concatenate( [x1,x2] ,axis = 0 )
    y = np.concatenate( [y1,y2], axis = 0 )


    X_ =np.ones((100*100,3))
    temp = np.linspace(-6,6,100)
    X_[:,0] = np.concatenate( [temp for i in range(100) ]  )
    X_[:,1] = [ i for i in temp  for j in range(100)]

    
    plot.figure()
    plot_data(x1,y1,x2,y2)

    y_ = [  np.concatenate( [1000*y1,y2] ) , np.concatenate( [y1,1000*y2] ), np.concatenate( [(N1+N2)/N1*y1 , (N1+N2)/N2*y2] ) ]
    for i in range(3):

        name = "y1:%.1f,y2:%.1f"%(y_[i][0], y_[i][N1] )
        p = MSE( X , y_[i]  )
        
        Z = p.g(X_)
        Z_ = Z.reshape((100,100))
        plot_plane( temp ,  Z_ , name , color = np.random.uniform( 0,0.7,3 ) )
    plot.savefig("mse.png",dpi=200)
    #plot.show()

