import numpy as np
import tensorflow as tf
import cv2

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

def to_grayscale(img):
    return np.dot(img[:,:,:3], [0.299, 0.587, 0.114])

def resize_img(img, size=(64, 64)):
    return cv2.resize(img, dsize=size).reshape(list(size) + [1])

def scale_color(img, max_val=255.):
    return img / max_val

def preprocess_atari(obs, size=(84, 84)):
    gray = to_grayscale(obs)
    resized = resize_img(gray, size)
    return scale_color(resized)