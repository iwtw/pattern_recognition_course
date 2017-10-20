import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import scipy.optimize as optimize
import sys
from utils import *

N1 = 440
N2 = 400
DIM = 2 + 1 
C = [ 1 , 10 , 1000 ]

class SVM:
    def __init__(self , X , y , C ):
        prod_X = np.zeros((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                prod_X[i,j] = X[i].T.dot(X[j])
        constraints = (
            {
                'type':'eq',
                'fun' : lambda lap : np.inner( lap,y ) ,
                'jac' : lambda lap : y
            },
        )
        def func(lap ,sign = 1.0):
            loss = 0.0
            for i in range( lap.shape[0] ):
                for j in range( lap.shape[0] ) :
                    loss += 0.5 * lap[i] * lap[j] * y[i] * y[j] * prod_X[i,j]
            loss -= np.sum( lap )
            return loss
        def func_deriv( lap , sign = 1.0):
            d = np.zeros_like( lap ) 
            for i in range( lap.shape[0] ):
                for j in range( lap.shape[0]):
                    d[i] += 0.5 * lap[j] * y[i] * y[j] * prod_X[i,j]
            d -=  1
            return d
        bounds = np.zeros((X.shape[0],2))
        bounds[:,1] = C * np.ones(X.shape[0])
        res = optimize.minimize( func , np.zeros( X.shape[0] ), jac = func_deriv , constraints = constraints ,  bounds = bounds , method = "SLSQP",  options= {'disp':True})
        self.lap = res.x
        self.sv_idx = np.flatnonzero( self.lap > 1e-10 )
        sv = X[self.sv_idx]
        y_sv = y[self.sv_idx]

        self.w =  np.sum(  np.reshape ( self.lap[self.sv_idx] * y_sv , (sv.shape[0],1) ) *  sv , axis = 0 ) 
        self.w0 = -1.0/( sv.shape[0] ) * np.sum ( sv.dot( self.w )  , axis = 0  )

    def g(self,x):
        return x.dot(self.w) + self.w0

if __name__ == "__main__" :
    x1 , y1 ,  x2 , y2  = generate_data(N1,N2, DIM)
    X = np.concatenate( [x1,x2] ,axis = 0 )
    y = np.concatenate( [y1,y2], axis = 0 )

    #print(svm.lap)

    temp = np.linspace(-6,6,100)
    X_ = np.ones( ( 100*100 ,3 ) )
    X_[:,0] = np.concatenate( [temp for i in range(100) ]  )
    X_[:,1] = [ i for i in temp  for j in range(100)]


    svm = []
    Z_ = []
    for c in C:
        plot.figure()
        color1 , color2 = plot_data( x1 ,y1, x2 ,y2)

        svm.append(  SVM( X , y , C = c ) )
        sv_idx1 = svm[-1].sv_idx [ np.flatnonzero( svm[-1].sv_idx < N1) ] 
        sv_idx2 = svm[-1].sv_idx [ np.flatnonzero( svm[-1].sv_idx >= N1 ) ] 
        plot.scatter( X[sv_idx1, 0 ] , X[sv_idx1,1] ,marker = "x" , color=color1  )
        plot.scatter( X[sv_idx2 , 0] , X[sv_idx2,1] , marker ='x' , color=color2 )
        
        Z_.append( svm[-1].g(X_ ) )
        Z_[-1] = Z_[-1].reshape((100,100)  )

        plot_plane( temp , Z_[-1] , name = "C:%d"%(c) )
        plot.savefig("svm_C:{}_with_sv.png".format(c))


    plot.figure()
    plot_data(x1,y1,x2,y2)
    for i in range(len(C)):
        plot_plane( temp , Z_[i]  , name = "C:%d"%(C[i])  )
    plot.savefig("svm.png")


   # plot.show()
