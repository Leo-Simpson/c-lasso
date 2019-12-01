import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import numpy.linalg as LA
from time import time
from CLasso.compact_func import Classo,pathlasso
n_lam = 100

def build_subset(n,nS): return(rd.permutation(n)[:nS])

def build_submatrix(matrix,subset):
    (A,C,y) = matrix
    subA,suby = A[subset],y[subset]
    return((subA,C,suby))


def biggest_indexes(L,q):
    qbiggest = []
    for i1 in range(q):
        indice = np.argmax(L)
        L[indice]=0
        qbiggest.append(indice)
    return(qbiggest)

def non_nul_indices(array):
    L = []
    for i in range(len(array)):
        if not (array[i]==0.):L.append(i)
    return(L)











def stability(matrix,SSmethod = 'first',numerical_method = "ODE",
              lam = 0.1,hd = False, q = 10 ,B = 50, pourcent_nS = 0.5 ,
              formulation = 'LS',plot_time=True, seed = 1 ):
    
    rd.seed(seed)    

    t0 = time()
    n, d = len(matrix[2]), len(matrix[0][0])
    nS = int(pourcent_nS*n)
    distribution=np.zeros(d)
    
    
    if (SSmethod == 'first') : 
    
    
    
        for i in range(B):
            subset = build_subset(n,nS)
            submatrix = build_submatrix(matrix,subset)
            # compute the path until n_active = q, and only take the last Beta
            beta = pathlasso(submatrix,n_active=q,lamin=0,
                             typ=formulation, meth = numerical_method,
                             plot_time=False,plot_sol=False,plot_sigm=False )[0][-1]
            qfirst = non_nul_indices(beta)
            for i in qfirst:
                distribution[i]+=1
    
    
    
    elif (SSmethod == 'lam') : 
    
        for i in range(B):
            subset = build_subset(n,nS)
            submatrix = build_submatrix(matrix,subset)
            regress = Classo(submatrix,lam,typ = formulation,
                             meth=numerical_method,plot_time=False,
                             plot_sol=False,plot_sigm=False)
            if (problem  in ['Concomitant','Concomitant_Huber']): beta =[0]
            else : beta = regress
            qbiggest = biggest_indexes(abs(beta),q)
            for i in qbiggest:
                distribution[i]+=1
                
    
    
    
    elif (SSmethod == 'max') : 
        
        
        # !!!!! A faire !!!!!!!!
        
        for i in range(B):
            subset = build_subset(n,nS)
            submatrix = build_submatrix(matrix,subset)
            # compute the path until n_active = q, and only take the last Beta
            beta = pathlasso(submatrix,n_active=q,lamin=0,
                             typ=formulation,meth = numerical_method,
                             plot_time=False,plot_sol=False,plot_sigm=False )[0][-1]
            qfirst = non_nul_indices(beta)
            for i in qfirst:
                distribution[i]+=1           
                
                
    
    
    if (plot_time): print("Running time : ", round(time()-t0,3))
    return(distribution * 1./B)









