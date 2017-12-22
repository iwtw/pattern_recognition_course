import numpy as np
def generate_data(num_samples , num_dims , mean , stddev):
    mean = np.array(mean)
    stddev = np.array(stddev)

    mean = mean + np.zeros(num_dims)
    stddev = stddev + np.zeros(num_dims)

    x = np.zeros((num_samples , num_dims),dtype = np.float32)
    for i in range(num_dims):
        x[:,i] = np.random.normal(mean[i],stddev[i],size = num_samples)
    return x


