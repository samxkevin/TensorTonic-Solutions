import numpy as np

def leaky_relu(x, a=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x=np.array(x,dtype=float)
    x[x<=0]*=a
    return x