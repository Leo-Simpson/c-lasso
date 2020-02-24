import numpy as np
import numpy.linalg as LA

from CLasso.solve_LS import problem_LS, algo_LS, pathalgo_LS
from CLasso.solve_Huber import problem_Huber, algo_Huber, pathalgo_Huber
from CLasso.solve_Concomitant import problem_Concomitant, algo_Concomitant, pathalgo_Concomitant
from CLasso.solve_Concomitant_Huber import problem_Concomitant_Huber, algo_Concomitant_Huber, pathalgo_Concomitant_Huber
from CLasso.path_alg import solve_cl_path, pathalgo_cl, solve_huber_cl_path, pathalgo_huber_cl, h_lambdamax

'''
Classo and pathlasso are the main functions, they can call every algorithm acording to the method and formulation required
'''

# can be 'Path-Alg', 'P-PDS' , 'PF-PDS' or 'DR'

def Classo(matrix,lam,typ = 'LS', meth='DR', rho = 1.345, get_lambdamax = False, true_lam=False, e=1., rho_classification=-1.):
    if(typ=='Concomitant'):
        if not meth in ['Path-Alg', 'DR']: meth='DR'
        pb = problem_Concomitant(matrix,meth,e=e)
        if (true_lam): beta,s = algo_Concomitant(pb,lam/pb.lambdamax)
        else : beta, s = algo_Concomitant(pb, lam)
        s = s/np.sqrt(e)

    elif(typ=='Concomitant_Huber'):
        if not meth in ['Path-Alg', 'DR']: meth='DR'
        pb  = problem_Concomitant_Huber(matrix,meth,rho,e=e)
        if (true_lam): beta,s = algo_Concomitant_Huber(pb,lam/pb.lambdamax,e=e)
        else : beta, s = algo_Concomitant_Huber(pb, lam,e=e)


    elif(typ=='Huber'):
        if not meth in ['Path-Alg', 'P-PDS' , 'PF-PDS' , 'DR']: meth = 'ODE'
        pb = problem_Huber(matrix,meth,rho)
        if (true_lam): beta = algo_Huber(pb,lam/pb.lambdamax)
        else : beta = algo_Huber(pb, lam)

    elif (typ == 'Huber_Classification'):
        if (true_lam):  BETA = solve_huber_cl_path(matrix, lam, rho_classification)[0] #TO DO HERE !!!!!!!!!
        else :    BETA = solve_huber_cl_path(matrix, lam, rho_classification)[0]
        beta = BETA[0]

    elif (typ == 'Classification'):
        if(true_lam) : BETA = solve_cl_path(matrix, lam)[0] # TO DO HERE !!!!!!!!
        else : BETA = solve_cl_path(matrix, lam)[0]
        beta = BETA[0]


    else: # LS
        if not meth in ['Path-Alg', 'P-PDS' , 'PF-PDS' , 'DR']: meth='DR'
        pb = problem_LS(matrix,meth)
        if (true_lam) : beta = algo_LS(pb,lam/pb.lambdamax)
        else : beta = algo_LS(pb,lam)

    if (typ  in ['Concomitant','Concomitant_Huber']): 
        if (get_lambdamax): return(pb.lambdamax,beta,s)
        else              : return(beta,s)
    if (get_lambdamax): return(pb.lambdamax,beta)
    else              : return(beta)


def pathlasso(matrix,lambdas=False,n_active=False,lamin=1e-2,typ='LS',meth='Path-Alg',rho = 1.345, true_lam = False, e= 1.,return_sigm= False,rho_classification=-1):
    if (type(lambdas)!= bool):
        if (lambdas[0]<lambdas[-1]): lambdas = [lambdas[i] for i in range(len(lambdas)-1,-1,-1)]  # reverse the list if needed
    else: lambdas = np.linspace(1.,lamin,100)

    if(typ=='Huber'):
        pb = problem_Huber(matrix,meth,rho)
        lambdamax = pb.lambdamax
        #if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        BETA  = pathalgo_Huber(pb,lambdas,n_active=n_active)

    elif(typ=='Concomitant'):
        pb = problem_Concomitant(matrix,meth,e=e)
        lambdamax = pb.lambdamax
        #if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        BETA,S = pathalgo_Concomitant(pb,lambdas,n_active=n_active)
        S=np.array(S)/np.sqrt(e)

    elif(typ=='Concomitant_Huber'):
        meth='DR'
        pb = problem_Concomitant_Huber(matrix,meth,rho)
        lambdamax = pb.lambdamax
        #if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        BETA,S = pathalgo_Concomitant_Huber(pb,lambdas,n_active=n_active)
        
    elif(typ == 'Huber_Classification'):
        lambdamax = h_lambdamax(matrix[0],matrix[2],rho)
        #if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        BETA = pathalgo_huber_cl(matrix, lambdas, rho_classification, n_active=n_active)

    elif (typ == 'Classification'):
        lambdamax = 2*LA.norm((matrix[0].T).dot(matrix[2]),np.infty)
        #if (true_lam): lambdas = [lamb / lambdamax for lamb in lambdas]
        BETA = pathalgo_cl(matrix, lambdas,n_active=n_active)

    else:
        pb = problem_LS(matrix,meth)
        lambdamax = pb.lambdamax
        #if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        BETA = pathalgo_LS(pb,lambdas,n_active=n_active)

    real_path = [lam*lambdamax for lam in lambdas]
    if(typ in ['Concomitant','Concomitant_Huber'] and return_sigm): return(BETA,real_path,S)
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