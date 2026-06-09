def he_initialization(W, fin):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    import numpy as np
    l=np.sqrt(6/fin)
    W=np.array(W,dtype=float)
    return 2*l*W-l