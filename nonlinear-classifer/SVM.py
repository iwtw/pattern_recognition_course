import sklearn.svm
import numpy as np
import time
from utils import * 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plot
import copy as copy

NUM_CLASSES = 2 
NUM_SAMPLES = 2000
NUM_DIMS = 2 
C = [0.3,0.5,1,3,5,8,10,30,50,100,500,1000,3000]
#C = [0.3,0.5,1,3,5,8,10]
KERNEL_LIST = ['linear','poly','rbf']#linear, polynomial , radial basis functions

def get_data():
    x1 = np.concatenate( [ generate_data( int(NUM_SAMPLES/2) , NUM_DIMS , mean = [-3,0] , stddev = 1 )  , generate_data( int(NUM_SAMPLES/2) , NUM_DIMS , mean = [-0.5,2] , stddev = 1) ] , axis = 0 ) 
    x2 = generate_data( NUM_SAMPLES , NUM_DIMS , mean = [0,-2] , stddev = 1 )

    shuffle_idx  = np.random.shuffle( np.array(range(NUM_SAMPLES*NUM_CLASSES)))
    x = np.concatenate([x1,x2],axis=0)
    y = np.concatenate([np.ones(NUM_SAMPLES,dtype = np.int8) , 2*np.ones(NUM_SAMPLES,dtype=np.int8)],axis=0)
    return x,y
if __name__ =="__main__":

    x_train , y_train = get_data()
    x_val , y_val = get_data()

    

    svm_list = []

    figno = 4
    best_svm = None
    best_kernel = None
    best_c = None
    best_err = 1
    for kernel in KERNEL_LIST:
        err_train_list , err_val_list , num_sv_list , time_list= [] , [] , [], []
        for c in C:
            s_time = time.time()
            svm = sklearn.svm.SVC( C = c ,  kernel = kernel )
            svm.fit( x_train , y_train )
            err_train = np.mean( svm.predict(x_train) != y_train )
            err_val = np.mean( svm.predict(x_val) != y_val )
            num_sv = np.sum( np.array(svm.n_support_) )
            err_train_list.append( err_train )
            err_val_list.append( err_val )
            num_sv_list.append( num_sv )
            e_time = time.time()
            time_list.append(e_time - s_time)
            if( err_val < best_err ):
                best_err = err_val
                best_kernel = kernel
                best_c = c 
                best_svm = svm
            print( "kernel:%s , c:%d , err_train %.2f%% , err_val %.2f%% , num_SV : %d , time cost: %.1f"%(kernel,c,100*err_train,100*err_val, num_sv , e_time-s_time ) )

        plot.figure(0)
        plot.plot( range(len(C)) , num_sv_list  , label = kernel )
        plot.figure(1)
        plot.plot( range(len(C)) , 100* np.array( err_train_list)  , label = kernel )
        plot.figure(2)
        plot.plot( range(len(C)) , 100* np.array( err_val_list)  , label = kernel )
        plot.figure(3)
        plot.plot( range(len(C)) , time_list , label = kernel )

    plot.figure(0)
    plot.title("numbers of support vectors using different C and kernels")
    plot.legend()
    plot.xticks( range(len(C)) , C )
    plot.xlabel("C")
    plot.ylabel("number of support vectors")
    plot.savefig("svm_sv.png",dpi=300)


    plot.figure(1)
    plot.title("train error using different C and kernels")
    plot.legend()
    plot.xticks( range(len(C)) , C )
    plot.xlabel("C")
    plot.ylim( 0,10 )
    plot.ylabel("train error(%)")
    plot.savefig("svm_train_error.png",dpi=300)

    plot.figure(2)
    plot.title("val error using different C and kernels")
    plot.legend()
    plot.xticks( range(len(C)) , C )
    plot.xlabel("C")
    plot.ylim( 0,10 )
    plot.ylabel("validation error(%)")
    plot.savefig("svm_val_error.png",dpi=300)

    plot.figure(3)
    plot.title("time cost using different C and kernels")
    plot.legend()
    plot.xticks( range(len(C)) , C )
    plot.xlabel("C")
    plot.ylabel("seconds")
    plot.savefig("svm_time_cost.png",dpi=300)
    



    plot.figure(4)
    plot.axis( (-8,4,-6,6))

    sv_idx = best_svm.support_
    lag =  best_svm.dual_coef_.squeeze()#lagrangian
    y_train_ = np.reshape( copy.copy( y_train ) , (-1,1))
    y_train_[ y_train==1 ] = 1  
    y_train_[ y_train==2 ] = -1 
    
    w0 = 0
    def K(x,y):
        return np.exp( -1.0/NUM_DIMS * np.linalg.norm( x-y , 2 , axis=1)**2  )
    def g(x_ref,x):
        return np.sum( lag * K( x_ref , x   ), axis = 0  ) + w0
        
        
    w0 = -1.0 * g( x_train[sv_idx] , x_train[sv_idx]) 
    x_axis = np.linspace( -8 , 8 , 100  )
    y_axis = np.linspace( -8 , 8 , 100 )
    X_ = np.ones((100*100,2))
    X_[:,0] = np.concatenate( [ x_axis for i in range(100) ] )
    X_[:,1] = np.array( [ i for i in y_axis for j in range(100) ] )

    Z = np.zeros(( X_.shape[0]))
    for i in range( len(Z) ):
        Z[i] = g(x_train[sv_idx],np.reshape( X_[i] ,(1,NUM_DIMS) ))

    idx_1 = np.flatnonzero( y_train==1 )
    idx_2 = np.flatnonzero( y_train==2 )
    plot.scatter( x_train[idx_1 , 0 ] , x_train[idx_1,1]   , label = "class 1" )
    plot.scatter( x_train[idx_2,0 ] , x_train[idx_2,1] , label="class 2" )

    plot.contour( x_axis , y_axis , np.reshape( Z , (100,100) )  , levels = [0] )
    plot.legend()

    plot.title("kernel {} c {}".format( best_kernel , best_c ))
    plot.savefig("kernel_{}_c_{}.png".format( best_kernel , best_c ))

