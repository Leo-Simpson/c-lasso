import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import numpy.linalg as LA
from time import time
from CLasso.compact_func import Classo,pathlasso
n_lam = 100

def indicator(BETA):
    l,k = len(BETA),len(BETA[0])
    IND = np.zeros((l,k))
    for i in range(l):
        for j in range(k):
            if (BETA[i][j]!=0. ): IND[i,j]=1.
    return IND


def selected_param(distribution,threshold):
    selected  = [False]*len(distribution)
    for i in range(len(distribution)):
        if (distribution[i] > threshold): selected[i]=True
    return(selected)


def build_subset(n,nS): return(rd.permutation(n)[:nS])

def build_submatrix(matrix,subset):
    (A,C,y) = matrix
    subA,suby = A[subset],y[subset]
    return((subA,C,suby))




def non_nul_indices(array):
    L = []
    for i in range(len(array)):
        if not (array[i]==0.):L.append(i)
    return(L)


def biggest_indexes(array,q):
    qbiggest = []
    nonnul = non_nul_indices(array)
    reduc_array = array[nonnul]
    for i1 in range(q):
        if not np.any(nonnul): break
        reduc_index = np.argmax(reduc_array)
        index = nonnul[reduc_index]
        if (reduc_array[reduc_index]==0.): break
        reduc_array[reduc_index]=0
        qbiggest.append(index)
    return(qbiggest)







def stability(matrix,SSmethod = 'first',numerical_method = "ODE",
              lam = 0.1,hd = False, q = 10 ,B = 50, pourcent_nS = 0.5 ,
              formulation = 'LS',plot_time=True, seed = 1, rho=1.345,
              true_lam = False):
    
    rd.seed(seed)    

    t0 = time()
    n, d = len(matrix[2]), len(matrix[0][0])
    nS = int(pourcent_nS*n)
    distribution=np.zeros(d)
    
    
    if (SSmethod == 'first') : 
    
        NN = 500
        lambdas= np.linspace(1.,0.,NN)
        distr_path = np.zeros((NN,d))
        for i in range(B):
            subset = build_subset(n,nS)
            submatrix = build_submatrix(matrix,subset)
            # compute the path until n_active = q, and only take the last Beta
            BETA = pathlasso(submatrix,lambdas=lambdas,n_active=q+1,lamin=0,
                             typ=formulation, meth = numerical_method,
                             plot_time=False,plot_sol=False,plot_sigm=False, rho = rho )[0]
            distr_path = distr_path + indicator(BETA)
        distribution = distr_path[-1]
        if (plot_time): print("Running time : ", round(time()-t0,3))
        return(distribution * 1./B, distr_path * 1./B,lambdas)
    
    elif (SSmethod == 'lam') : 
    
        for i in range(B):
            subset = build_subset(n,nS)
            submatrix = build_submatrix(matrix,subset)
            regress = Classo(submatrix,lam,typ = formulation,
                             meth=numerical_method,plot_time=False,
                             plot_sol=False,plot_sigm=False, rho = rho, true_lam = true_lam)
            if (formulation  in ['Concomitant','Concomitant_Huber']): beta =regress[0]
            else : beta = regress
            qbiggest = biggest_indexes(abs(beta),q)
            for i in qbiggest:
                distribution[i]+=1
                
    
    
    
    elif (SSmethod == 'max') : 
        
       
        
        for i in range(B):
            subset = build_subset(n,nS)
            submatrix = build_submatrix(matrix,subset)
            # compute the path until n_active = q, and only take the last Beta
            BETA = pathlasso(submatrix,n_active=False,lamin=1e-2,
                             typ=formulation,meth = numerical_method,
                             plot_time=False,plot_sol=False,plot_sigm=False, rho = rho )[0]
            betamax = np.amax( abs(np.array(BETA)), axis = 0 )
            qmax = biggest_indexes(betamax,q)
            for i in qmax:
                distribution[i]+=1           
                
                
    
    
    if (plot_time): print("Running time : ", round(time()-t0,3))
    return(distribution * 1./B)






