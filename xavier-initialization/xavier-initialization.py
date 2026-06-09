def xavier_initialization(W, fin, fout):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    import numpy as np
    W=np.array(W,dtype=float)
    l=np.sqrt(6/(fin+fout))
    W=W*2*l-l
    return W