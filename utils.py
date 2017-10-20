import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot

def generate_data( N1 , N2 ,DIM ):
    x1 = np.ones( ( N1 , DIM ) )
    x2 = np.ones ( ( N2 , DIM ) )

    x1[:,0] = -1.7 + 1.1 * np.random.randn(N1)
    x1[:,1] = 1.6 + 0.9 * np.random.randn(N1)

    x2[:,0] = 1.3 + 1.0 * np.random.randn(N2)
    x2[:,1] = -1.5 + 0.8 * np.random.randn(N2)
    
    y1 = np.ones(N1)
    y2 = -1 * np.ones(N2)
    return x1 , y1 ,  x2 , y2

def plot_plane( temp , Z ,   name , color = np.random.uniform(0 , 0.7 ,3) ):
    ct = plot.contour(temp,temp,Z,levels=[0])
    ct.collections[0].set_label(name)
    ct.collections[0].set_color(color)
    plot.legend()
    

def plot_data( x1 , y1 , x2 , y2 ):
    plot.axis( (-6,6,-6,6) )
    color1 = plot.scatter(x1[:,0],x1[:,1],marker=".").get_facecolor()
    color2 = plot.scatter(x2[:,0],x2[:,1],marker=".").get_facecolor()
    return color1 , color2
    

