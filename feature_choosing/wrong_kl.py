import sklearn.decomposition
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plot
import copy as copy

NUM_DIMS_TRANSFORMED = 2 
NUM_DIMS = 6
NUM_CLASSES = 3
N = 500
STDDEV = [
        [0.8,0.9,0.7,1.5,1.4,1.1],
        [0.9,0.7,0.8,1.2,1.1,1.0],
        [0.7,0.8,0.9,1.3,1.2,1.3] 
        ]
MEAN = [
        [-2,-1,2,-1,-1,-2],
        [1,-2,-2,1,-1,-1],
        [-2,2,-1,-1,1,1]
        ]


def generate_data():
    x = np.zeros( (N*NUM_CLASSES , NUM_DIMS ) , dtype = np.float32 )
    y = np.ones(  N*NUM_CLASSES   , dtype = np.int8 )
    for i in range( NUM_CLASSES ):  
        y[i*N:(i+1)*N] *= i

    for i in range( NUM_CLASSES ):
        x[i*N:(i+1)*N] = np.random.normal( MEAN[i] , STDDEV[i] , size = (N,NUM_DIMS) )
        
    return x,y


if __name__=="__main__":
    x , y = generate_data()
    pca = sklearn.decomposition.PCA(NUM_DIMS_TRANSFORMED)
    x_ = pca.fit_transform(x,y)
    
    cls_idx = []
    for i in range(NUM_CLASSES):
        cls_idx.append( np.flatnonzero( y == i  ) )
    plot.figure()
    for i in range(NUM_CLASSES):
        plot.scatter( x_[cls_idx[i],0] , x_[cls_idx[i],1] )
    plot.savefig("sklearn_PCA_transformed.png",dpi=200)

    sigma = np.zeros( (NUM_DIMS , NUM_DIMS) )
    for i in range(NUM_CLASSES):
        miu_i = np.mean( x[cls_idx[i]] , axis = 0 )
        t = x[cls_idx[i]] - miu_i
        #t = x[cls_idx[i]]
        sigma_i = 1.0/(N-1) * ( t.T.dot(t) ) 
        sigma += 1.0/NUM_CLASSES * sigma_i
    print(sigma)

    w , v = np.linalg.eig( sigma  )
    sorted_idx = np.argsort( w )

    T = v[:,sorted_idx][:,-NUM_DIMS_TRANSFORMED:]
   # print(w)
   # print(v)
   # print(T)
    x_ = copy.copy(x)
    #for i in range(NUM_CLASSES):
    #    x_[cls_idx[i]] -= np.mean(x[cls_idx[i]] , axis = 0)
    x_ -= np.mean(x_)
    x_ = x_.dot(T)
    #x_ = x - np.mean(x , axis = 0 )
    #x_ = x_.dot(T) 
    #x_ = x.dot(T)
    plot.figure()
    for i in range(NUM_CLASSES):
        plot.scatter( x_[cls_idx[i],0] , x_[cls_idx[i],1] )
    plot.savefig("wrong_kl_transformed.png",dpi=200)


