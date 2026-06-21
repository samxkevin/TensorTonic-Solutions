import numpy as np

def adam_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, e=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    p=np.array(p,dtype=float)
    g=np.array(g,float)
    m=np.array(m,dtype=float)
    v=np.array(v,dtype=float)
    t=np.array(t,dtype=float)
    e=np.array(e,dtype=float)
    mn=b1*m+(1-b1)*g
    vn=b2*v+(1-b2)*np.square(g)
    mt=mn/(1-b1**t)
    vt=vn/(1-b2**t)
    pn=p-lr*(mt/(np.sqrt(vt)+e))
    return pn,mn,vn