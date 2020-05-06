import numpy as np
import numpy.linalg as LA

from .solve_R1 import problem_R1, Classo_R1, pathlasso_R1
from .solve_R2 import problem_R2, Classo_R2, pathlasso_R2
from .solve_R3 import problem_R3, Classo_R3, pathlasso_R3
from .solve_R4 import problem_R4, Classo_R4, pathlasso_R4
from .path_alg import solve_path, pathalgo_general, h_lambdamax

'''
Classo and pathlasso are the main functions, they can call every algorithm acording to the method and formulation required
'''

# can be 'Path-Alg', 'P-PDS' , 'PF-PDS' or 'DR'

def Classo(matrix,lam,typ = 'R1', meth='DR', rho = 1.345, get_lambdamax = False, true_lam=False, e=1., rho_classification=-1.):
    if(typ=='R3'):
        if not meth in ['Path-Alg', 'DR']: meth='DR'
        pb = problem_R3(matrix,meth,e=e)
        if (true_lam): beta,s = Classo_R3(pb,lam/pb.lambdamax)
        else : beta, s = Classo_R3(pb, lam)
        s = s/np.sqrt(e)

    elif(typ=='R4'):
        if not meth in ['Path-Alg', 'DR']: meth='DR'
        pb  = problem_R4(matrix,meth,rho,e=e)
        if (true_lam): beta,s = Classo_R4(pb,lam/pb.lambdamax,e=e)
        else : beta, s = Classo_R4(pb, lam,e=e)


    elif(typ=='R2'):
        if not meth in ['Path-Alg', 'P-PDS' , 'PF-PDS' , 'DR']: meth = 'ODE'
        pb = problem_R2(matrix,meth,rho)
        if (true_lam): beta = Classo_R2(pb,lam/pb.lambdamax)
        else : beta = Classo_R2(pb, lam)

    elif (typ == 'C2'):
        if (true_lam):  BETA = solve_path(matrix, lamin, False, rho_classification, 'huber_cl')[0] #TO DO HERE !!!!!!!!!
        else :    BETA = solve_path(matrix, lamin, False, rho_classification, 'huber_cl')[0]
        beta = BETA[0]

    elif (typ == 'C1'):
        if(true_lam) : BETA = solve_path(matrix,lamin, False,0, 'cl')[0] # TO DO HERE !!!!!!!!
        else : BETA = solve_path(matrix,lamin, False,0, 'cl')[0]
        beta = BETA[0]


    else: # LS
        if not meth in ['Path-Alg', 'P-PDS' , 'PF-PDS' , 'DR']: meth='DR'
        pb = problem_R1(matrix,meth)
        if (true_lam) : beta = Classo_R1(pb,lam/pb.lambdamax)
        else : beta = Classo_R1(pb,lam)

    if (typ  in ['R3','R4']):
        if (get_lambdamax): return(pb.lambdamax,beta,s)
        else              : return(beta,s)
    if (get_lambdamax): return(pb.lambdamax,beta)
    else              : return(beta)


def pathlasso(matrix,lambdas=False,n_active=0,lamin=1e-2,typ='LS',meth='Path-Alg',rho = 1.345, true_lam = False, e= 1.,return_sigm= False,rho_classification=-1):
    Nactive = n_active
    if(Nactive==0):Nactive=False
    if (type(lambdas)!= bool):
        if (lambdas[0]<lambdas[-1]): lambdas = [lambdas[i] for i in range(len(lambdas)-1,-1,-1)]  # reverse the list if needed
    else: lambdas = np.linspace(1.,lamin,100)

    if(typ=='R2'):
        pb = problem_R2(matrix,meth,rho)
        lambdamax = pb.lambdamax
        if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        BETA  = pathlasso_R2(pb,lambdas,n_active=Nactive)

    elif(typ=='R3'):
        pb = problem_R3(matrix,meth,e=e)
        lambdamax = pb.lambdamax
        if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        BETA,S = pathlasso_R3(pb,lambdas,n_active=Nactive)
        S=np.array(S)/np.sqrt(e)

    elif(typ=='R4'):
        meth='DR'
        pb = problem_R4(matrix,meth,rho)
        lambdamax = pb.lambdamax
        if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        BETA,S = pathlasso_R4(pb,lambdas,n_active=Nactive)
        
    elif(typ == 'C2'):
        lambdamax = h_lambdamax(matrix[0],matrix[2],rho)
        if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        BETA = pathalgo_general(matrix, lambdas, 'huber_cl', n_active=Nactive, rho=rho_classification)

    elif (typ == 'C1'):
        lambdamax = 2*LA.norm((matrix[0].T).dot(matrix[2]),np.infty)
        if (true_lam): lambdas = [lamb / lambdamax for lamb in lambdas]
        BETA = pathalgo_general(matrix, lambdas, 'cl', n_active=Nactive)

    else:
        pb = problem_R1(matrix,meth)
        lambdamax = pb.lambdamax
        if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        BETA = pathlasso_R1(pb,lambdas,n_active=n_active)

    real_path = [lam*lambdamax for lam in lambdas]
    if(typ in ['R3','R4'] and return_sigm): return(BETA,real_path,S)
    return(BETA,real_path)
 
    


'''
# Cost fucntions for the three 'easiest' problems. Useful for test, to compare two solutions slightly different
def L_LS(A,y,lamb,x): return(LA.norm( A.dot(x) - y )**2 + lamb * LA.norm(x,1))
def L_conc(A,y,lamb,x): return(LA.norm( A.dot(x) - y ) + np.sqrt(2)*lamb * LA.norm(y,1))
def L_H(A,y,lamb,x,rho): return(hub( A.dot(x) - y , rho) + lamb * LA.norm(x,1))

def hub(r,rho) : 
    h=0
    for j in range(len(r)):
        if(abs(r[j])<rho): h+=r[j]**2
        elif(r[j]>0)     : h+= (2*r[j]-rho)*rho
        else             : h+= (-2*r[j]-rho)*rho
    return(h)
'''