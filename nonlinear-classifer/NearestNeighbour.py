import numpy as np
import copy as copy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plot
from utils import *


class MutiEdit:
    def __init__(self,x,y,num_folds,num_iters,x_val=None , y_val=None):
        x = copy.copy( x )
        y = copy.copy( y )
        figno = 0 
        while True:
            
            done = False
            for it in range( num_iters):
                shuffle_idx = np.random.shuffle(np.array( range(x.shape[0]) ) )
                x = x[shuffle_idx].squeeze()
                y = y[shuffle_idx].squeeze()
                m = int(np.floor( x.shape[0] / num_folds ))
                correct_idx = []
                exists_wrong = False
                for i in range (num_folds):
                    x_ref = np.delete( x , range(i*m,(i+1)*m) , axis=0 )
                    y_ref = np.delete( y , range(i*m ,(i+1)*m) , axis = 0)
                    for j in range( i*m,(i+1)*m ):
                        diff = np.reshape(x[j],(1,-1)) - x_ref
                        diff **= 2
                        diff = np.sum( diff , axis = 1  )
                        diff **= 0.5
                        if y_ref[np.argmin( diff )] == y[ j ]:
                            correct_idx.append( j )
                        else:
                            exists_wrong = True
                if not exists_wrong:
                    done = True
                    break
                else:
                    correct_idx = np.array(correct_idx)
                    x = x[correct_idx].squeeze()
                    y = y[correct_idx].squeeze()
            figno +=1
            plot.figure()
            plot.title("reference set {}".format(figno))
            plot.axis((-8,4,-6,6))
            idx_1 = np.flatnonzero( y==1)
            idx_2 = np.flatnonzero( y==2)
            print( idx_1.shape[0] , idx_2.shape[0] )
            plot.scatter(x[idx_1,0],x[idx_1,1],label ="class 1")
            plot.scatter(x[idx_2,0],x[idx_2,1],label="class 2")
            plot.legend()
            plot.savefig("ME_ref{}.png".format(figno),dpi=300)
            
            if done:
                break
        self.x_ref = x
        self.y_ref = y 
    def predict(self,x):
        y_pred = np.zeros( x.shape[0]  )
        for i in range( len(x) ):
            diff = np.reshape( x[i] , (1,-1) ) - self.x_ref 
            diff **=2
            diff = np.sum( diff , axis = 1 )
            y_pred [ i ]= self.y_ref [ np.argmin( diff ,axis = 0  )  ]
        return y_pred

class Condensing:
    def __init__(self,x,y):
        x_store = x[0].reshape((1,-1))
        y_store = y[0].reshape( 1 )
        x_grabbag = x[1:].reshape( (-1,x.shape[1] ))
        y_grabbag = y[1:]
        figno = 1
        while(True):
            x_store = x_store.reshape((-1,x.shape[1]))
            x_grabbag = x_grabbag.reshape(( -1,x.shape[1] ))
            wrong_idx = []
            for i in range( y_grabbag.shape[0] ):
                diff = x_grabbag[i]  - x_store
                diff**=2
                diff = np.sum( diff , axis = 1 )
                j = np.argmin( diff ,axis = 0  )
                if ( y_grabbag[i] != y_store[ j ] ):
                    x_store = np.insert( x_store , len( x_store) , x_grabbag[i], axis =0  )
                    y_store = np.insert( y_store , len(y_store) , y_grabbag[i] , axis=0 )
                    wrong_idx.append(i)
            x_grabbag = np.delete( x_grabbag , wrong_idx , axis = 0 )
            y_grabbag = np.delete( y_grabbag , wrong_idx , axis = 0 )

            plot.figure()
            plot.title( "Condensing : reference set {}".format( figno ) )
            plot.axis((-8,4,-6,6))
            idx_1 = np.flatnonzero( y_store == 1 )
            idx_2 = np.flatnonzero( y_store == 2 )
            print(len(idx_1) , len(idx_2))
            plot.scatter( x_store[ idx_1 , 0] , x_store[idx_1 ,1] , label ="class 1"  )
            plot.scatter( x_store[ idx_2 , 0] , x_store[idx_2 , 1 ] , label="class 2" )
            plot.legend()
            plot.savefig( "condensing_ref%d.png"%(figno) , dpi = 300 )
            figno +=1

            if len(wrong_idx) == 0  :
                break
        self.x_ref = x_store
        self.y_ref = y_store
                    
    def predict(self , x  ):
        y_pred = np.zeros( x.shape[0] )
        for i in range( x.shape[0] ):
            diff = np.reshape( x[i] , (1,-1) ) - self.x_ref  
            diff **=2
            diff = np.sum(diff,axis=1)
            y_pred[i] = self.y_ref [ np.argmin( diff , axis = 0 ) ]
        return y_pred
            

    

NUM_SAMPLES = 2000
NUM_DIMS = 2 
NUM_CLASSES = 2 
def get_data():
    x1 = np.concatenate( [ generate_data( int(NUM_SAMPLES/2) , NUM_DIMS , mean = [-3,0] , stddev = 1 )  , generate_data( int(NUM_SAMPLES/2) , NUM_DIMS , mean = [-0.5,2] , stddev = 1) ] , axis = 0 ) 
    x2 = generate_data( NUM_SAMPLES , NUM_DIMS , mean = [0,-2] , stddev = 1 )

    x = np.concatenate([x1,x2],axis=0)
    y = np.concatenate([np.ones(NUM_SAMPLES,dtype = np.int8) , 2*np.ones(NUM_SAMPLES,dtype=np.int8)],axis=0)
    return x,y

if __name__ =="__main__":
    x_train , y_train = get_data()
    x_val , y_val = get_data()

    plot.figure()
    plot.title("x original")
    plot.axis((-8,4,-6,6))
    idx_1 = np.flatnonzero( y_train==1 )
    idx_2 = np.flatnonzero( y_train==2 )
    plot.scatter( x_train[idx_1 , 0 ] , x_train[idx_1,1]   , label = "class 1" )
    plot.scatter( x_train[idx_2,0 ] , x_train[idx_2,1] , label="class 2" )
    plot.legend()
    plot.savefig("ME_original_x.png",dpi=300)

    
    me = MutiEdit(x_train,y_train,5,1)
    print( "Muti-edit , acc_train : %.2f%% , acc_val %.2f%%"%( 100*np.mean ( y_train == me.predict( x_train )) , 100*np.mean( y_val == me.predict( x_val )) ) ) 

    condensing = Condensing( x_train , y_train )
    print("Condensing , acc_train  %.2f%% , acc_val %.2f%%"%(100*np.mean( y_train == condensing.predict( x_train )) , 100*np.mean( y_val == condensing.predict( x_val) )  ))
    

    plot.figure()
    plot.title("decision plane")
    plot.axis((-8,4,-6,6))
    idx_1 = np.flatnonzero( y_train==1 )
    idx_2 = np.flatnonzero( y_train==2 )
    plot.scatter( x_train[idx_1 , 0 ] , x_train[idx_1,1] )
    plot.scatter( x_train[idx_2,0 ] , x_train[idx_2,1]  )

    x_axis = np.linspace( -8 , 8 , 100 )
    y_axis = np.linspace( -8 , 8 , 100 )
    X = np.zeros((100*100,2))
    X[:,0] = np.concatenate( [ x_axis for i in range(100) ] )
    X[:,1] = np.array( [ i for i in y_axis for j in range(100) ] )
    
    z = np.zeros( (100,100) )
    def plot_plane(x,model,label ):
        idx_1 = np.flatnonzero( model.y_ref == 1 )
        idx_2 = np.flatnonzero( model.y_ref == 2 )
        dis_1 = np.zeros( ( 100*100 )  )
        dis_2 = np.zeros( (100*100  ) )
        for i in range(  100*100 )  :
            dis_1[i]  = np.min( np.sum( ( np.reshape( x[i] , (1,-1) )- model.x_ref[idx_1] )**2 ,axis = 1 ) )
            dis_2[i] = np.min(np.sum( ( np.reshape( x[i] ,(1,-1) ) - model.x_ref[idx_2])**2 , axis = 1 ))
      #  for i in range( 100):
      #      for j in range( 100 ):
      #          z[i,j] = x[i,j,0] - x[i,j,1]

      #  print(z.shape)
      #  ct = plot.contour ( x_axis , y_axis ,  z  )
        ct = plot.contour( x_axis , y_axis , np.reshape( dis_1 - dis_2 , (100,100) ) , levels = [0] )
        ct.collections[0].set_label( label )
        ct.collections[0].set_color( np.random.uniform( 0,1,3 ) )

    plot_plane( X,me , "Multi edit" )
    plot_plane( X,condensing , "Condensing" )
    plot.legend()
    plot.savefig( "NeareastNeighbour_decision_plane.png",dpi=300 )


