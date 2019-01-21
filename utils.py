import numpy as np

def reshape_train_var(var):
    var_shape = np.array(var[0]).shape
    if var_shape == ():
        return var
    n_samples = len(var)
    concated = np.concatenate(var)
    reshaped = np.array(concated).reshape([n_samples] + list(var_shape))
    return reshaped