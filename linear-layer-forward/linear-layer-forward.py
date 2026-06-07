def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    import numpy as np
    W=np.array(W)
    b=np.array(b)
    y=np.dot(X,W)+b
    return y.tolist()