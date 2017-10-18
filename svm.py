import numpy as np
import matplotlib.pyplot as plot
import scipy.optimize as optimize
import sys

N1 = 12
N2 = 10
DIM = 2 + 1 

class SVM:
    def __init__(self , X , y , C ):
        prod_X = np.zeros((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                prod_X[i,j] = X[i].T.dot(X[j])
        constraints = (
            {
                'type':'eq',
                'fun' : lambda lap : lap.T.dot(y) ,
                'jac' : lambda lap : np.sum(y)
            },
            {
                'type':'ineq',
                'fun' : lambda lap : lap,
                'jac' : lambda lap : np.ones_like(lap)
            },
            {
                'type':'ineq',
                'fun' : lambda lap : -1.0 * ( lap - C ),
                'jac' : lambda lap : -1.0 * np.ones_like( lap )
            }
        )
        def func(lap ,sign = 1.0):
            loss = 0.0
            for i in range( lap.shape[0] ):
                for j in range( lap.shape[0] ) :
                    loss += 0.5 * lap[i] * lap[j] * y[i] * y[j] * prod_X[i,j]
            loss -= np.sum( lap )
            return loss
        def func_deriv( lap , sing = 1.0):
            d = np.zeros_like( lap ) 
            for i in range( lap.shape[0] ):
                for j in range( lap.shape[1]):
                    d[i] += 0.5 * lap[j] * y[i] * y[j] * prod_X[i,j]
            d -= np.ones_like( d ) 
            return d
        res = optimize( func , np.zeros( X.shape[0] ), jac = func_deriv , constraints = constraints ,  method = 'SLSQP' , options= {'disp':True})
        self.lap = res.x
        printf(self.lap)
        
        
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
    x1 , y1 , x2 , y2 = generate_data()
    plot.axis((-6,6,-6,6))
    plot.scatter(x1[0,:],x1[1,:],marker=".")
    plot.scatter(x2[0,:],x2[1,:],marker=".")
    X = np.concatenate( [x1,x2] , axis = 1 )
    X = X.T
    svm = SVM( X , y , C = 1.0 )
    print(svm.lap)
