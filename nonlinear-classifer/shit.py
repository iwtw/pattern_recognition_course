import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plot
import numpy as np

HIDDEN_SIZES = [ 3,5,8,10,15,20,25,35,50,100,300,500,800 ]
train_err = [ 3,5,8,10,15,20,25,35,50,100,300,500,800 ]
val_err = [ 3,5,8,10,15,20,25,35,50,100,300,500,800 ]
if __name__ =="__main__":
    plot.figure()
    plot.title("training error vs validation error")
    plot.plot( range(len( HIDDEN_SIZES)) , train_err, label='train err'  )
    plot.plot( range(len(HIDDEN_SIZES)) , val_err , label ='val err' )
    plot.legend()
    plot.xlabel("hidden size")
    plot.ylabel("error(%)")
    plot.xticks(range(len(HIDDEN_SIZES)),HIDDEN_SIZES)
    plot.savefig( "summary.png" )
        
        #for it in 

    
