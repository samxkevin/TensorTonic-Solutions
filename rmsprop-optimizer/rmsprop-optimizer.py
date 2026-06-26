import numpy as np

def rmsprop_step(w, g, s, lr=0.001, b=0.9, e=1e-8):
    w=np.array(w,dtype=float)
    g=np.array(g,dtype=float)
    s=np.array(s,dtype=float)
    st=s*b+(1-b)*g**2
    w=w-(lr/(np.sqrt(st)+e))*g
    return w,st