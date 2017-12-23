import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

class KMeans:
    def __init__(self  ):
        pass
    def cluster(self  , x , k , max_iters = int(1e6)  , eps = 1e-6 ):
        mean = x[ np.random.choice( x.shape[0] , k , replace = False ) ]
        n , d = x.shape
        for it in range( max_iters ):
            prev_mean = mean
            diff = x.reshape( (n,1,d) ) - mean.reshape( ( 1 , k , d ) )
            dis = np.linalg.norm( diff ,  ord = 2 , axis = 2   )
            labels = np.argmin( dis , axis = 1  )
            for i in range(k):
                cls_idx =  np.flatnonzero( labels == i )
                mean[i] = np.mean( x[cls_idx] , axis = 0  )
            if np.linalg.norm( mean - prev_mean , ord = 2 ) < eps:
                break
        return labels



NUM_CLASSES = 3
N = 500
NUM_DIMS=3

STDDEV = [ [0.8,0.9,0.7] 
        , [0.9,0.7,0.8] 
        , [0.7,0.8,0.9] ]
MEAN = [ [-2,-1,2]
        ,[1,-2,-2]
        ,[-2,2,-1] ]

K = 3

def generate_data():
    x = np.zeros( (N*NUM_CLASSES , NUM_DIMS ) , dtype = np.float32 )
    y = np.ones(  N*NUM_CLASSES   , dtype = np.int8 )
    for i in range( NUM_CLASSES ):  
        y[i*N:(i+1)*N] *= i

    for i in range( NUM_CLASSES ):
        x[i*N:(i+1)*N] = np.random.normal( MEAN[i] , STDDEV[i] , size = (N,NUM_DIMS) )
        
    return x,y

if __name__=="__main__":
    x,y = generate_data()

    fig = plot.figure()
    plot_3d = Axes3D(fig)
    for i in range(NUM_CLASSES):
        cls_idx = np.flatnonzero( y == i ) 
        plot_3d.scatter( x[cls_idx, 0] , x[cls_idx,1] , x[cls_idx,2] )
    plot.savefig( "orginal.png" , dpi=200 )


    cluster = KMeans()

    y_ = cluster.cluster( x , K , 1000 )
    fig = plot.figure()
    plot_3d = Axes3D(fig)
    for i in range(K):
        cls_idx = np.flatnonzero( y_ == i )
        plot_3d.scatter( x[cls_idx,0] , x[cls_idx,1] , x[cls_idx,2])
    plot.savefig( "k-means.png",dpi=200 )

