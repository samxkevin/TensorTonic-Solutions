import numpy as np

def adamw_step(w, m, v, g, lr=0.001, b1=0.9, b2=0.999, wd=0.01, e=1e-8):
    """
    Performing one AdamW update step.
    """
    w=np.array(w,dtype=float)
    g=np.array(g,float)
    m=np.array(m,dtype=float)
    v=np.array(v,dtype=float)
    wd=np.array(wd,dtype=float)
    e=np.array(e,dtype=float)
    mt=m*b1+(1-b1)*g
    vt=v*b2+(1-b2)*g**2
    return w-lr*(wd*w+(mt/(np.sqrt(vt)+e))),mt,vt