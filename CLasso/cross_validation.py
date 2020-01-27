import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import numpy.linalg as LA
from CLasso.compact_func import Classo,pathlasso
n_lam = 500

def compute_1SE(mse_max,MSE,i): 
    j=i
    while(MSE[j]<mse_max): j-=1
    return j


def train_test_CV(n,k,test_pourcent):
    idx, training_size = rd.permutation(n), int(n-n*test_pourcent)
    idx_train, idx_test = idx[:training_size], idx[training_size:]
    SUBLIST,end = [],0 
    for i in range(k):
        begin,end = end,end+training_size//k
        if(i<training_size%k): end+=1
        SUBLIST.append(idx[begin:end])
    return(SUBLIST, idx_train,idx_test) 


def train_test_i (SUBLIST,i):
    training_set,test_set = np.array([],dtype=int),SUBLIST[i]
    for j in range(len(SUBLIST)):
        if (j != i): training_set = np.concatenate((training_set,SUBLIST[j]))
    return(training_set,test_set)
            

def training(matrices,typ,num_meth, training_set, rho, e,lambdas):
    (A,C,y)   = matrices
    mat       = (A[training_set],C,y[training_set]) 
    sol       = pathlasso(mat,lambdas = lambdas,typ=typ,meth=num_meth,
                            plot_time=False,plot_sol=False,plot_sigm=False,rho = rho, e=e)[0]
    return(sol)


def test_i (matrices,typ,num_meth, SUBLIST,i, rho, e,lambdas):
    training_set,test_set = train_test_i (SUBLIST,i)
    BETA                  = training(matrices,typ,num_meth, training_set, rho, e, lambdas)
    n_lam = len(lambdas)
    residual = np.zeros(n_lam)
    for j in range(n_lam):
        residual[j] = accuracy_func(matrices[0][test_set],matrices[2][test_set],BETA[j],typ)/len(test_set)
    return(residual)

def average_test(matrices,typ,num_meth, SUBLIST, rho, e,lambdas):
    k = len(SUBLIST)
    n_lam = len(lambdas)
    RESIDUAL = np.zeros((k,n_lam))
    for i in range(k):
        RESIDUAL[i,:] = test_i (matrices,typ,num_meth, SUBLIST,i, rho, e, lambdas)
    MSE = np.mean(RESIDUAL,axis = 0)
    SE = np.std(RESIDUAL,axis = 0) # official standard error should be divided by sqrt(k) ... 
    return(MSE,SE)

def CV(matrices,k,typ='LS',num_meth="ODE",test=0., seed = 1, rho = 1.345, e= 1.,lambdas = np.linspace(1.,1e-3,500),oneSE = True):
    
    rd.seed(seed)
    (A,C,y) = matrices
    SUBLIST, idx_train, idx_test = train_test_CV(len(y),k,test)
    MSE,SE  = average_test(matrices,typ,num_meth, SUBLIST, rho, lambdas) 
    i       = np.argmin(MSE)
    lam     = lambdas[i]
    i_1SE   = compute_1SE(MSE[i]+SE[i],MSE,i)
    lam_1SE = lambdas[i_1SE]
    if oneSE : 
        out = Classo((A[idx_train],C,y[idx_train]),lam_1SE,typ=typ,meth=num_meth,plot_time=False,plot_sol=False,plot_sigm=False, rho=rho, e=e)
    else :     
        out = Classo((A[idx_train],C,y[idx_train]),lam,typ=typ,meth=meth,plot_time=False,plot_sol=False,plot_sigm=False, rho=rho, e=e)
    return(out,MSE,SE,i,i_1SE)
    
        
# Cost fucntions for the three 'easiest' problems. 
def hub(r,rho) : 
    h=0
    for j in range(len(r)):
        if(abs(r[j])<rho): h+=r[j]**2
        elif(r[j]>0)     : h+= (2*r[j]-rho)*rho
        else             : h+= (-2*r[j]-rho)*rho
    return(h)

def accuracy_func(A,y,beta, typ='LS',rho = 1.345):
    if (typ == 'Huber'):    return(hub( A.dot(beta) - y , rho * LA.norm(y,np.infty)/np.sqrt(len(y))) )
    else :                  return(LA.norm( A.dot(beta) - y )**2)
