import numpy as np
import tensorflow as tf

def reshape_train_var(var):
    var_shape = np.array(var[0]).shape
    if var_shape == ():
        return var
    n_samples = len(var)
    concated = np.concatenate(var)
    reshaped = np.array(concated).reshape([n_samples] + list(var_shape))
    return reshaped

def gaussian_likelihood(x, mu, std):
    """
    Calculate the Gaussian Likelihood of a given input for using in continuous
    action spaces.
    """
    pre_sum = -(0.5*tf.log(2.*np.pi)) - (0.5*tf.log(std)) - (tf.square(x - mu))/(2.*std+1e-8)
    
    return tf.reduce_sum(pre_sum, axis=1)