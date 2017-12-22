import numpy as np

EPS = 1e-5
class GA:
    def __init__(self, length , population_size = 500  , initial_population = None  ):
        self.population = initial_population
        if (self.population == None):
            self.population = np.random.uniform( 0 , 1 , (population_size,length) )
            self.population[self.population >=0.5 ] = 1
            self.population[self.population < 0.5 ] = 0

        if  self.population.ndim != 2 :
            self.population = np.reshape( self.population , (-1,self.population.shape[0]) )
        self.population = np.array( self.population , dtype = np.int8 )
    def fitness( self , x , y , k  ):
        #根据 x,y计算得的可分性 以及 要挑选的特征个数k计算适应度
        #fitness[i] = 1( 染色体[i]的特征个数= k) * J
        #其中J为可分性判据
        max_y = np.max(y)
        min_y = np.min(y)
        cls_idx = []
        for i in range( min_y , max_y +1 ):
            cls_idx.append( np.flatnonzero( y == i ) )
        num_classes = len( cls_idx )

        #先验概率p
        p = np.zeros( num_classes )
        for i in range(num_classes):
            p[i] = len( cls_idx[i] ) / len(x) 


        ret = np.zeros( len(self.population) )
        for idx in range( len(self.population) ):
            features_idx = np.flatnonzero( self.population[idx] )
            if (len(features_idx) != k ):
                ret[idx] = 0
                continue
            xx = x[:,features_idx]
            #均值mean
            mean = np.zeros( (num_classes,k) )
            for i in range(num_classes):
                mean[i] = np.mean( xx[cls_idx[i]] , axis = 0 )
            mean_total = np.mean( xx  , axis = 0 )

            Sw = np.zeros( (k,k) )
            for i in range( num_classes ):
                for j in range( len(cls_idx[i]) ):
                    t = np.reshape( xx[cls_idx[i][j]] - mean[i] , (k,1))
                    Sw += p[i] * 1/len(cls_idx[i]) * t.dot(t.T)

            Sb = np.zeros( (k,k) )
            for i in range( num_classes):
                temp = np.reshape( mean[i] - mean_total ,(k,1))
                Sb += p[i] * temp.dot(temp.T) 

            
            J1 = np.trace( np.linalg.inv(Sw).dot(Sb)  )
           # print(np.linalg.det(Sb), np.linalg.det(Sw) )
            #J2 = np.log( np.linalg.det(Sb) / np.linalg.det(Sw) )
            #J3 = np.trace(Sb) / np.trace( Sw )
            #J4 = np.trace(Sb+Sw) / np.trace(Sw)
            
            ret[idx] = J1
            #print(Sb,Sw)
            #print(J1,J2,J3,J4)

        return ret

    def evolve( self , num_iters , combination_rate , mutation_rate , x , y ,  k  ):
        prev_avg_fitness = 0
        for it in range( num_iters ):
            #选择复制
            fit = self.fitness( x,y,k )
            avg_fit = np.mean( fit )
            print(it,avg_fit)
            if( ( np.abs( avg_fit - prev_avg_fitness ) )< EPS):
                break
            prev_avg_fitness = avg_fit
            fit /= np.sum(fit)
            acc_fit = np.zeros_like( fit )
            for i in range( acc_fit.shape[0] ):
                acc_fit[i] = acc_fit[i-1] + fit[i]
            new_population = np.zeros_like( self.population ,dtype = np.int8)
            for i in range( new_population.shape[0] ):
                dice = np.random.uniform(0,1)
                for j in range( acc_fit.shape[0] ):
                    if dice <= acc_fit[j]:
                        new_population[i] = self.population[j]
                        break

            self.population = new_population

            for i in range( self.population.shape[0] ):
                for j in range( self.population.shape[1]  ):
                    dice = np.random.uniform( 0,1 )
                    #交叉
                    if dice < combination_rate:
                        ii = int(np.floor( np.random.uniform( 0 , self.population.shape[0] ) ))

                        temp = self.population[i][j]
                        self.population[i][j]  = self.population[ii][j]
                        self.population[ii][j] = temp
                    dice = np.random.uniform( 0 , 1 )
                    #变异
                    if dice < mutation_rate:
                        self.population[i][j] ^= 1 
            avg_fitness = np.average( self.fitness( x,y,k ) )

NUM_ITERS = 1000
MUTATION_RATE = 0.01
COMBINATION_RATE = 0.5

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
        y[i*N:(i+1)*N] *= i+1

    for i in range( NUM_CLASSES ):
        x[i*N:(i+1)*N] = np.random.normal( MEAN[i] , STDDEV[i] , size = (N,NUM_DIMS) )
        
    return x,y
if __name__ =="__main__":
    x , y = generate_data()
    #for i in range( 6):
    #    ga = GA( length = i+1 ,  initial_population = np.ones( i+1 , dtype  = np.int8 ) )
    #    ga.fitness(x,y,i+1)
    ga = GA( length = NUM_DIMS  )
    ga.evolve( NUM_ITERS , COMBINATION_RATE , MUTATION_RATE ,  x , y , 3 )
    print(ga.population[:20])

