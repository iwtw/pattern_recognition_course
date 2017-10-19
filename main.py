import numpy as np
import matplotlib.pyplot as plot
from perceptron import Perceptron
from mse import MSE
from svm import SVM

N1 = 440
N2 = 400
DIM = 2 + 1 

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

def plot_plane( temp , Z ,  name , color ):
    Z = Z.reshape( ( temp.shape[0] , temp.shape[0])  )
    ct = plot.contour( temp,temp,Z,levels=[0] )
    ct.collections[0].set_label('hyperplance by ' + name)
    ct.collections[0].set_color(color)
    plot.legend()

if __name__ == "__main__" :
    x1 , y1 ,  x2 , y2  = generate_data()
    plot.axis( (-6,6,-6,6) )
    color1 = plot.scatter(x1[:,0],x1[:,1],marker=".").get_facecolor()
    color2 =  plot.scatter(x2[:,0],x2[:,1],marker=".").get_facecolor()

    perceptron = Perceptron()
    X = np.concatenate( [x1,x2] ,axis = 0 )
    y = np.concatenate( [y1,y2], axis = 0 )


    

    temp = np.linspace(-6,6,100)
    X_ = np.ones( ( 100*100 ,3 ) )
    X_[:,0] = np.concatenate( [temp for i in range(100) ]  )
    X_[:,1] = [ i for i in temp  for j in range(100)]
    

    perceptron = Perceptron()
    for it in range(1000):
        perceptron.update(X,y  )

    Z = perceptron.g( X_ )
    plot_plane( temp , Z , 'perceptron' , color = [0.5,0.2,0.8] )
    
    mse = MSE( X , y )
    Z = mse.g( X_ )
    plot_plane( temp , Z , 'MSE', color = [0.2,0.6,0.7])

    svm = SVM( X , y , C = 10.0 )
    Z = svm.g( X_ )
    plot_plane( temp , Z , 'SVM' , color = [0.3,0.2,0.3] )

    plot.show()
