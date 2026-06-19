import numpy as np
def gradient_descent_quadratic(a, b, c, x, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x=np.array(x,dtype=float)
    for i in range(0,steps):
        x=x-lr*(2*a*x+b)
    return x