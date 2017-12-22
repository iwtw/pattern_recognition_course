import torch
import torch.optim.lr_scheduler
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plot
from utils import *

NUM_CLASSES = 5#类别个数 
NUM_SAMPLES = 1000#每类的样本个数
NUM_DIMS = 3 
MEAN =[ [-2,-2,-2] , [2,-2,-2] , [-2,2,-2] , [-2,-2,2] , [2,2,2] ]
STDDEV = [ [0.8,0.9,0.7] , [0.9,0.9,0.8] , [0.9,0.8,0.7],[0.7,0.9,0.8],[1.0,0.9,0.9] ]

HIDDEN_SIZES = [ 3,4,5,6,7,8,10,15,20,50,100,300,500]#隐层神经元个数
NUM_EPOCHS = 20 
LEARNING_RATE = 1e-3

class NN(torch.nn.Module):
    def __init__(self,hidden_size):
        super(NN,self).__init__()
        self.fc1 = torch.nn.Linear(3 , hidden_size , bias = True )
        self.fc2 = torch.nn.Linear(hidden_size , NUM_CLASSES , bias = True )
    def forward(self,x):
        x = self.fc1(x)
        x = torch.sigmoid( self.fc2(x) )
        return x
    def loss(self,x,y):
        if(len(x.size())==1):
            n=1
        else:
            n=x.size()[0]
        y_ = self.forward( x )
        loss = torch.sum(( y_ - y)**2 )/ n
        return loss


if __name__=="__main__":

    x_list = []
    for i in range(NUM_CLASSES):
        x_list.append(generate_data(num_samples = NUM_SAMPLES , num_dims = NUM_DIMS , mean = MEAN[i] , stddev = STDDEV[i] )  )
    x = np.concatenate(x_list , axis = 0 )
    y = np.zeros( (NUM_SAMPLES * NUM_CLASSES , NUM_CLASSES ) , dtype = np.float32 )
    for i in range(NUM_CLASSES):
        y[i*NUM_SAMPLES:(i+1)*NUM_SAMPLES,i] += 1 


    train_idx = [ [ j for j in range( i*NUM_SAMPLES , int( (i+0.8)*NUM_SAMPLES  )) ]  for i in range(NUM_CLASSES)]
    val_idx = [ [ j for j in range( int( (i+0.8)*NUM_SAMPLES ), int((i+0.9)*NUM_SAMPLES )) ] for i in range(NUM_CLASSES)  ]
    test_idx = [ [ j for j in range(int ( (i+0.9)*NUM_SAMPLES ), int( (i+1)*NUM_SAMPLES )) ]  for i in range(NUM_CLASSES) ]
    def V(x):
        return torch.autograd.Variable( torch.from_numpy(x))
    train_idx = np.reshape( np.array( train_idx , dtype = np.int32 ) , -1 )
    val_idx = np.reshape( np.array( val_idx , dtype = np.int32 ) , -1 )
    test_idx = np.reshape( np.array( test_idx , dtype = np.int32 ) , -1 )
    x_train = V(x[train_idx])
    y_train = V(y[train_idx])
    x_val = V(x[val_idx])
    y_val = V(y[val_idx])
    x_test = V(x[test_idx])
    y_test = V(y[test_idx])

    
    best_acc_val = 0
    best_hidden_size = 0
    best_nn = None
    
    train_err =  np.zeros( len(HIDDEN_SIZES ))
    val_err = np.zeros( len(HIDDEN_SIZES )) 

    for i in range( len(HIDDEN_SIZES) ):
        nn = NN(HIDDEN_SIZES[i])
        optimizer = torch.optim.SGD(nn.parameters(), lr = LEARNING_RATE )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,5,gamma=0.5)
        print("hidden size : %d"%(HIDDEN_SIZES[i]))
        loss_train_history , loss_val_history , acc_val_history , acc_train_history = [] , [] , [] , []
        for epoch in range(NUM_EPOCHS+1):
            if(epoch>0):
                scheduler.step()
            for it in range(x_train.size()[0]):
                if epoch==0:
                    break
                loss =  nn.loss( x_train[it] , y_train[it] )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_val = nn.loss(x_val,y_val).data.numpy().squeeze()
            loss_train = nn.loss(x_train,y_train).data.numpy().squeeze()
            acc_train = np.mean ( np.argmax( nn.forward(x_train).data.numpy() , axis = 1 ) == np.argmax( y_train.data.numpy(), axis = 1  )  ) 
            acc_val = np.mean ( np.argmax( nn.forward(x_val).data.numpy() , axis = 1 ) == np.argmax( y_val.data.numpy(), axis = 1  )  ) 
            loss_train_history.append( loss_train )
            loss_val_history.append( loss_val )
            acc_train_history.append( acc_train )
            acc_val_history.append( acc_val )
            print("epoch%d , train_loss : %.4f , val_loss : %.4f , acc_train : %.2f%% , acc_val : %.2f%%"%( epoch ,loss_train,loss_val,acc_train*100,acc_val*100 ))

        if best_acc_val < acc_val:
            best_acc_val = acc_val
            best_nn = nn
            best_hidden_size = HIDDEN_SIZES[i]
        train_err[i] = 1 - acc_train 
        val_err[i] = 1 - acc_val

        plot.figure()
        plot.title("ANN training history of hidden size %d"%HIDDEN_SIZES[i])
        plot.plot( range(NUM_EPOCHS + 1 ) , loss_train_history , label="train loss"  )
        plot.plot( range(NUM_EPOCHS + 1 ) , loss_val_history , label="validation loss"  )
        plot.xlim( 0,20 )
        plot.xlabel( "epochs" )
        plot.legend()
        plot.savefig( "nn_size%d.png"%(HIDDEN_SIZES[i]),dpi=300 )


    plot.figure()
    plot.title("training error vs validation error")
    plot.plot( range(len(HIDDEN_SIZES)) , train_err*100, label='train err'  )
    plot.plot( range(len(HIDDEN_SIZES)) , val_err*100 , label ='val err' )
    plot.legend()
    plot.xlabel("hidden size")
    plot.ylabel("error(%)")
    plot.xticks(range(len(HIDDEN_SIZES)) , HIDDEN_SIZES )
    plot.savefig( "summary.png",dpi=300 )
    test_acc = np.mean ( np.argmax( best_nn.forward(x_test).data.numpy() , axis = 1 ) == np.argmax( y_test.data.numpy(), axis = 1  )  ) 
    print("best hidden size:%d , best_test_acc:%.4f"%(best_hidden_size,test_acc))
        
        #for it in 

    
