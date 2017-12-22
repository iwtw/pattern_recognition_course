
def generate_data( N1 , N2 ,DIM ):
    x1 = np.ones( ( N1 , DIM ) )
    x2 = np.ones ( ( N2 , DIM ) )

    x1[:,0] = -0.7 + 0.3 * np.random.randn(N1)
    x1[:,1] = 1.2 + 0.9 * np.random.randn(N1)

    x2[:,0] = 0.3 + 1.2 * np.random.randn(N2)
    x2[:,1] = -1.8 + 0.3 * np.random.randn(N2)
    
    y1 = np.ones(N1)
    y2 = -1 * np.ones(N2)
    return x1 , y1 ,  x2 , y2
