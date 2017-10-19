import numpy as np
import matplotlib.pyplot as plot
import scipy.optimize as optimize
import sys

N1 = 440
N2 = 400
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
        def func_deriv( lap , sing = 1.0):
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
       # print(self.lap)
       # print(np.inner( self.lap , y) )
        self.w = np.sum( np.reshape( self.lap * y,(X.shape[0],1) ) * X , axis = 0  )
        self.w0 = -1.0/(X.shape[0]) * np.sum ( X.dot( self.w )  , axis = 0  )

    def g(self,x):
        return x.dot(self.w) + self.w0
        
        
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
    color1 = plot.scatter(x1[:,0],x1[:,1],marker=".").get_facecolor()
    color2 = plot.scatter(x2[:,0],x2[:,1],marker=".").get_facecolor()
    X = np.concatenate( [x1,x2] ,axis = 0 )
    y = np.concatenate( [y1,y2], axis = 0 )

    svm = SVM( X , y , C = 10.0 )
    print(svm.lap)

    temp = np.linspace(-6,6,100)
    X_ = np.ones( ( 100*100 ,3 ) )
    X_[:,0] = np.concatenate( [temp for i in range(100) ]  )
    X_[:,1] = [ i for i in temp  for j in range(100)]
    Z = svm.g(X_ )
    Z = Z.reshape((100,100)  )
    sv_idx = np.nonzero( svm.lap > 1e-10 )[0]
    print(sv_idx)
    sv_idx1 = sv_idx [ np.nonzero( sv_idx <= N1) ] 
    sv_idx2 = sv_idx [ np.nonzero( sv_idx > N1 ) ] 
    plot.scatter( X[sv_idx1, 0 ] , X[sv_idx1,1] ,marker = "x" , color=color1  )
    plot.scatter( X[sv_idx2 , 0] , X[sv_idx2,1] , marker ='x' , color=color2 )
    plot.contour(  temp  , temp , Z , levels=[0] )
    plot.show()
