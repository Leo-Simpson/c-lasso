import numpy as np
import numpy.linalg as LA

N = 10000


def prox(b,w,zeros): return(np.minimum(b+w,zeros)+np.maximum(b-w,zeros)) 

def QQ(coef,A): return(coef*(A.T).dot(LA.inv(2*np.eye(A.shape[0])+coef*A.dot(A.T))),LA.inv(2*np.eye(A.shape[1])+coef*(A.T).dot(A)))    
    
   
    
def solve(matrices,lam):
    (A,C,y) = matrices
    m,d,k = len(y), len(A[0]), len(C) 
    
    mu = 1.95 - lam/2
    gam= d/(2*lam)
    c  =  gam/(mu-1)    
    
    
    AtAnorm = LA.norm(A.T.dot(A),2)
    tol, weights= 1e-7 * LA.norm(y)/AtAnorm ,np.ones(d)*LA.norm(A.T.dot(y),np.inf)/AtAnorm
    if (k==0): Proj = np.eye(d)
    else : Proj = np.eye(len(C[0]))-LA.multi_dot([C.T,np.linalg.inv(C.dot(C.T) ),C])
    b ,xbar,x = np.zeros(d),np.zeros(d),np.zeros(d)
    zerod = np.zeros(d)

    w  = (gam*lam)*weights

    Q1,Q2  = QQ(c/AtAnorm,A)
    qy_mult = Q1.dot(y)*(mu-1)
    for i in range(N):
        xbar= xbar + mu*(prox(2*b-xbar,w,zerod)-b)
        x   = x    + mu*(Proj.dot(2*b-x)       -b)
        
        nv_b = (2-mu)*b
        nv_b = nv_b + qy_mult + Q2.dot(x+xbar- 2*nv_b)
        if (i%2==1 and LA.norm(b-nv_b)<tol): return(b,i)       
        b = nv_b

    print('NO CONVERGENCE')
    return(b,'slow')  



def warmpath_solve(matrices,lamin,nlam=100):
    (A,C,y) = matrices
    m,d,k = len(y), len(A[0]), len(C) 
    
    mu = 1.7
    gam= d/2
    c  =  gam/(mu-1)
    
    
    AtAnorm = LA.norm(A.T.dot(A),2)
    tol, weights= 1e-7 * LA.norm(y)/AtAnorm ,np.ones(d)*LA.norm(A.T.dot(y),np.inf)/(AtAnorm)
    if (k==0): Proj = np.eye(d)
    else : Proj = np.eye(len(C[0]))-LA.multi_dot([C.T,np.linalg.inv(C.dot(C.T) ),C])
    b ,xbar,x, lb = np.zeros(d),np.zeros(d),np.zeros(d), []
    zerod = np.zeros(d)

    
    Q1,Q2  = QQ(c/AtAnorm,A)
    QA_mult = (np.eye(d)-2*Q2)*(2-mu)
    qy_mult = Q1.dot(y)*(mu-1)
    
    w  = (gam)*weights
    dw = w*(1-lamin)/nlam
    
    for j in range(nlam):
        w = w - dw
        for i in range(N):
            
            xbar = xbar + mu*(prox(2*b-xbar,w,zerod)-b)
            x    = x    + mu*(Proj.dot(2*b-x)       -b)
  
            nv_b = (2-mu)*b
            nv_b = nv_b + qy_mult + Q2.dot(x+xbar- 2*nv_b)
            if (i>1 and LA.norm(b-nv_b)<tol): break
            b = nv_b
        lb.append(b)
    return(np.array(lb))
