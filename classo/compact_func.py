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

def Classo(matrix,lam,typ = 'R1', meth='DR', rho = 1.345, get_lambdamax = False, true_lam=False, e=None, rho_classification=-1., w = None, intercept = False):

    if not w is None : matrices = (  matrix[0]/w, matrix[1]/w,matrix[2] )
    else : matrices = matrix
    if intercept : 
        means = (np.mean(matrices[0],axis=0), np.mean(matrices[2]) )
        matrices = (matrices[0]-means[0],matrices[1],matrices[2]-means[1])

    
    if(typ=='R3'):
        if not meth in ['Path-Alg', 'DR']: meth='DR'
        if e is None or e == len(matrices[0])/2: 
            r = 1.
            pb = problem_R3(matrices,meth)
        else: 
            r = np.sqrt(2*e/len(matrices[0]))
            pb = problem_R3((matrices[0]*r,matrices[1],matrices[2]*r),meth)
        lambdamax = pb.lambdamax
        if (true_lam): beta,s = Classo_R3(pb,lam/lambdamax)
        else : beta, s = Classo_R3(pb, lam)
        s = s/np.sqrt(e)

    elif(typ=='R4'):
        if not meth in ['Path-Alg', 'DR']: meth='DR'
        if e is None or e == len(matrices[0]): 
            r = 1.
            pb = problem_R4(matrices,meth,rho)
        else: 
            r = np.sqrt(e/len(matrices[0]))
            pb = problem_R4((matrices[0]*r,matrices[1],matrices[2]*r),meth,rho/r)

        lambdamax = pb.lambdamax
        if (true_lam): beta,s = Classo_R4(pb,lam/lambdamax)
        else : beta, s = Classo_R4(pb, lam)


    elif(typ=='R2'):
        if not meth in ['Path-Alg', 'P-PDS' , 'PF-PDS' , 'DR']: meth = 'ODE'
        pb = problem_R2(matrices,meth,rho)
        lambdamax = pb.lambdamax
        if (true_lam): beta = Classo_R2(pb,lam/lambdamax)
        else : beta = Classo_R2(pb, lam)

    elif (typ == 'C2'):
        lambdamax = h_lambdamax(matrices[0],matrices[2],rho)
        if true_lam : BETA = solve_path(matrices, lam/lambdamax, False, rho_classification, 'huber_cl')[0]
        else : BETA = solve_path(matrices, lam, False, rho_classification, 'huber_cl')[0]
        beta = BETA[-1]

    elif (typ == 'C1'):
        lambdamax = h_lambdamax(matrices[0],matrices[2],0)
        if (true_lam): BETA = solve_path(matrices,lam/lambdamax, False,0, 'cl')[0]
        else : BETA = solve_path(matrices,lam, False,0, 'cl')[0]
        beta = BETA[-1]


    else: # LS
        if not meth in ['Path-Alg', 'P-PDS' , 'PF-PDS' , 'DR']: meth='DR'
        pb = problem_R1(matrices,meth)
        lambdamax = pb.lambdamax
        if (true_lam) : beta = Classo_R1(pb,lam/lambdamax)
        else : beta = Classo_R1(pb,lam)


    if not w is None : beta = beta / w
    if intercept : 
        betaO = means[1] - means[0].dot(beta)
        beta = np.array([betaO]+list(beta))

    if (typ  in ['R3','R4']):
        if (get_lambdamax): return(lambdamax,beta,s)
        else              : return(beta,s)
    if (get_lambdamax): return(lambdamax,beta)
    else              : return(beta)


def pathlasso(matrix,lambdas=False,n_active=0,lamin=1e-2,typ='R1',meth='Path-Alg',rho = 1.345, true_lam = False, e= None,return_sigm= False,rho_classification=-1, w =None, intercept = False):

    
    Nactive = n_active
    if(Nactive==0):Nactive=False
    if (type(lambdas)!= bool):
        if (lambdas[0]<lambdas[-1]): lambdass = [lambdas[i] for i in range(len(lambdas)-1,-1,-1)]  # reverse the list if needed
        else : lambdass = [lambdas[i] for i in range(len(lambdas))]
    else: lambdass = np.linspace(1.,lamin,80)


    if not w is None : matrices = (  matrix[0]/w, matrix[1]/w,matrix[2] )
    else : matrices = matrix
    
    if intercept : 
        means = (np.mean(matrices[0],axis=0), np.mean(matrices[2]) )
        matrices = (matrices[0]-means[0],matrices[1],matrices[2]-means[1])

    if(typ=='R2'):
        pb = problem_R2(matrices,meth,rho)
        lambdamax = pb.lambdamax
        if (true_lam): lambdass=[lamb/lambdamax for lamb in lambdass]
        BETA  = pathlasso_R2(pb,lambdass,n_active=Nactive)

    elif(typ=='R3'):
        if e is None or e == len(matrices[0])/2: 
            r = 1.
            pb = problem_R3(matrices,meth)
        else: 
            r = np.sqrt(2*e/len(matrices[0]))
            pb = problem_R3((matrices[0]*r,matrices[1],matrices[2]*r),meth)
        lambdamax = pb.lambdamax
        if (true_lam): lambdass=[lamb/lambdamax for lamb in lambdass]
        BETA,S = pathlasso_R3(pb,lambdass,n_active=Nactive)
        S=np.array(S)/r**2
        BETA = np.array(BETA)


    elif(typ=='R4'):
        if e is None or e == len(matrices[0]): 
            r = 1.
            pb = problem_R4(matrices,meth,rho)
        else: 
            r = np.sqrt(e/len(matrices[0]))
            pb = problem_R4((matrices[0]*r,matrices[1],matrices[2]*r),meth,rho/r)

        lambdamax = pb.lambdamax
        if (true_lam): lambdass=[lamb/lambdamax for lamb in lambdass]
        BETA,S = pathlasso_R4(pb,lambdass,n_active=Nactive)
        S=np.array(S)/r**2
        BETA = np.array(BETA)
        
    elif(typ == 'C2'):
        lambdamax = h_lambdamax(matrices[0],matrices[2],rho)
        if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdass]
        BETA = pathalgo_general(matrices, lambdass, 'huber_cl', n_active=Nactive, rho=rho_classification)

    elif (typ == 'C1'):
        lambdamax = h_lambdamax(matrices[0],matrices[2],0)
        if (true_lam): lambdass = [lamb / lambdamax for lamb in lambdass]
        BETA = pathalgo_general(matrices, lambdass, 'cl', n_active=Nactive)

    else:
        pb = problem_R1(matrices,meth)
        lambdamax = pb.lambdamax
        if (true_lam): lambdass=[lamb/lambdamax for lamb in lambdass]
        BETA = pathlasso_R1(pb,lambdass,n_active=n_active)

    real_path = [lam*lambdamax for lam in lambdass]

    if not w is None : BETA = np.array([beta / (w+1e-3) for beta in BETA])
    if intercept : 
        BETA = np.array([ [means[1] - means[0].dot(beta) ]+list(beta) for beta in BETA] )


    if(typ in ['R3','R4'] and return_sigm): return(np.array(BETA),real_path,S)
    return(np.array(BETA),real_path)
 
    







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


'''
Remark about R3 : 

    The functions Classo_R3 and pathlasso_R3 only allows to compute the solution of R3, with e = n/2 : 
         min_(beta,sigma) ||Xbeta - y||^2/sigma + n/2 sigma + lambda ||beta||1 = L'(X,y,lambda,beta,sigma) with C.b= 0 and sigma > 0 
    
    Now let's say that we have to solve the same problem but with e = r^2 * n/2 :
        L(r,X,y,lamba,beta,s)   = ||Xbeta - y||^2/s + r^2 *n/2 s + lambda ||beta||1
                                = ||(Xr)beta - yr||^2/sigma + n/2 sigma + lambda ||beta||1      with sigma = s*r^2
                                = L'(Xr,yr,lambda,beta,s*r^2)
        so we just have to solve the problem with e=n/2 , r = sqrt(2e/n), X = X*r , y = y*r and then divide sigma by r^2
 




The same remark with R4 holds, but with e = n instead. 



    The functions Classo_R4 and pathlasso_R4 only allows to compute the solution of R4, with e = n/2 : 
         min_(beta,sigma) h_rho(Xbeta - y / sigma) sigma + n sigma + lambda ||beta||1 = L'(X,y,rho,lambda,beta,sigma) with C.b= 0 and sigma > 0 
    
    Now let's say that we have to solve the same problem but with e = r * n :
        L(r,X,y,rho,lamba,b,s)  = h_rho((Xbeta - y)/s)  s  + n r^2 s + lambda   ||b||1
                                = h_p  ( (Xbeta - y)/(sr))  r^2 s  + n r^2 s + lambda ||b||1                    because h_rho(x) =  h_p (x/r) * r^2 , with p = rho/r
                                = h_p  ( ((Xr)beta - yr)/sigma)  sigma  + n sigma + lambda ||b||1      with sigma = s*r^2 
                                = L'(Xr,yr,rho/r, lambda,beta,s*r^2)

        so we just have to solve the problem with e=n , r = sqrt(2e/n), X = X*r , y = y*r, rho = rho/r and then divide sigma by r^2
 




'''