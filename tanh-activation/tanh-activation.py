import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x=np.array(x,dtype=float)
    t=(np.exp(x)-np.exp(-1*x))/(np.exp(x)+np.exp(-1*x))
    return t