import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
import numpy.linalg as LA
from CLasso.compact_func import Classo,pathlasso
n_lam = 100

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
            

def training(matrices,typ,lamin, training_set):
    (A,C,y)   = matrices
    mat       = (A[training_set],C,y[training_set]) 
    return(pathlasso(mat,lamin=lamin,typ=typ,meth='ODE',plot_time=False,plot_sol=False,plot_sigm=False)[0])


def test_i (matrices,typ,lamin, SUBLIST,i):
    training_set,test_set = train_test_i (SUBLIST,i)
    BETA                  = training(matrices,typ,lamin, training_set)
    L = np.zeros(n_lam)
    for j in range(n_lam):
        L[j] = accuracy_func(matrices[0][test_set],matrices[2][test_set],BETA[j],typ)
    return(L)

def average_test(matrices,typ,lamin, SUBLIST):
    AVG = np.zeros(n_lam)
    for i in range(len(SUBLIST)):
        L = test_i (matrices,typ,lamin, SUBLIST,i)
        AVG=AVG+L
    return(AVG)

def CV(matrices,k,typ='LS',test=0.4,lamin=1e-2):
    (A,C,y) = matrices
    SUBLIST, idx_train, idx_test = train_test_CV(len(y),k,test)
    AVG1  = average_test(matrices,typ,lamin, SUBLIST) 
    LAM = np.linspace(1.,lamin,n_lam)
    Davg = (AVG1[2]-AVG1[-1])
    AVG2 = AVG1 - LAM*Davg
#     plt.plot(LAM,AVG1),plt.xlabel("lambda"),plt.ylabel("||y-Ax||2"),plt.title('AVG1'),plt.show()
#     plt.plot(LAM,AVG2),plt.xlabel("lambda"),plt.ylabel("||y-Ax||2"),plt.title('AVG2'),plt.show()
    i1 = np.argmin(AVG1)
    i2 = np.argmin(AVG2)
    lam1 = LAM[i1]
    lam2 = LAM[i2]
    print('lam =',lam1)
    beta1 = Classo((A[idx_train],C,y[idx_train]),lam1,typ=typ,plot_time=False,plot_sol=False,plot_sigm=False)
    beta2 = Classo((A[idx_train],C,y[idx_train]),lam2,typ=typ,plot_time=False,plot_sol=False,plot_sigm=False)
    return(beta1)
    
        
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
