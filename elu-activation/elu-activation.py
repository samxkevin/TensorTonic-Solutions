import numpy as np
def elu(x, a):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    for i in range(0,len(x)):
        if x[i]<=0:
            x[i]=(np.exp(x[i])-1)*a
    return x