import numpy as np
import sklearn.decomposition
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  as plot

x = [[0,0,0],[1,0,0],[1,0,1],[1,1,0],[0,0,1],[0,1,0],[0,1,1],[1,1,1]]
y = [0,0,0,0,1,1,1,1]

pca = sklearn.decomposition.PCA(2)
x_ = pca.fit_transform(x,y)
plot.scatter( x_[0:4,0] , x_[0:4,1] )
plot.scatter( x_[4:,0] , x_[4:,1] )
plot.savefig("test.png",dpi=200)
print(x_)

