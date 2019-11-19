import numpy as np
import numpy.linalg as LA
    
'''    
Problem    :   min h_rho((Ab - y)/sigma)sigma + simga + lambda ||b||1 with C.b= 0, sigma>0 

Dimensions :   A : m*d  ;  y : m  ;  b : d   ; C : k*d

The first function compute a solution of a Lasso problem for a given lambda. The parameters are lam (lambda/lambdamax, \in [0,1]) and pb, which has to be a 'problem_LS type', which is defined bellow in order to contain all the important parameters of the problem. One can initialise it this way : pb = class_of_problem.problem(data=(A,C,y),type_of_algo). We solve the problem without normalizing anything. 
'''    


def algo_Concomitant_Huber(pb,lam):
    
    pb_type = pb.type      # 2prox
    (m,d,k),(A,C,y)  = pb.dim,pb.matrix 
    lamb,rho  = lam * pb.lambdamax, pb.rho

    regpath = pb.regpath
    Anorm   = LA.norm(A,'fro')
    tol     = pb.tol * LA.norm(y) # tolerance rescaled
    Proj    = proj_c(C,d)   # Proj = I - C^t . (C . C^t )^-1 . C 
    gamma   = LA.norm(y)*pb.gam / (Anorm**2)
    w,zerod = lamb *gamma*pb.weights, np.zeros(d) # two vectors usefull to compute the prox of f(b)= sum(wi |bi|)
    mu, c   = pb.mu, pb.c
    Q1,Q2   = QQ(c,A) 
    QA,qy   = Q1.dot(A), Q1.dot(y)
    root    = [0.]*len(y)
    proj_sigm = lambda vect: ([sum(vect)/len(vect)]*len(vect))
    xs,nu,o,xbar,x = pb.init
    #2prox
    print(pb.sigmax)
    if (pb_type == '2prox'):
        
        for i in range(pb.N):
            nv_b, nv_s = x + Q1.dot(o) - QA.dot(x) - Q2.dot(x-xbar), (xs+nu)/2
            if (i>0 and LA.norm(b-nv_b)*Anorm +LA.norm(s-nv_s)<2*tol):
                if (regpath):return(b,(xs,nu,o,xbar,x),sum(s)/len(s)/pb.sigmax) 
                else :       return(b,sum(s)/len(s))
                        
            s,b = nv_s, nv_b
            Ab = A.dot(b)
            p1,p2,root = prox_phi_2(xs,2*Ab-o-y,gamma/c,root,rho)
            sup = [ proj_sigm(nu)-s , p1-s , p2 + y - Ab , prox(2*b-xbar,w,zerod)-b, Proj.dot(2*b-x)-b]
            xs,nu,o,xbar,x = xs+mu*sup[0] ,  nu+mu*sup[1] ,  o+mu*sup[2] ,  xbar+mu*sup[3] ,  x+mu*sup[4]
            if (LA.norm(b)+LA.norm(s)>1e6): 
                print('DIVERGENCE')
                return(b,np.sqrt(m)*sum(s)/len(s))
        print('NO CONVERGENCE')
        return(b,sum(s)/len(s))
    
    
    
    
    
    print('none of the cases ! ')        
    
    

    
'''
This function compute the the solution for a given path of lam : by calling the function 'algo' for each lambda with warm start, or wuth the method ODE, by computing the whole path thanks to the ODE that rules Beta and the subgradient s, and then to evaluate it in the given finite path.  
'''
    
def pathalgo_Concomitant_Huber(pb,path):
    n = pb.dim[0]
    BETA,SIGMA,tol = [],[],pb.tol

    save_init = pb.init   
    pb.regpath = True
    for lam in path:
        X = algo_Concomitant_Huber(pb,lam)     
        BETA.append(X[0]), SIGMA.append(X[2])
        pb.init = X[1]    
        if(sum([ (abs(X[0][i])>1e-2) for i in range(len(X[0])) ])>=n):
                pb.init, BETA = save_init, BETA + [BETA[-1]]*(len(path)-len(BETA))
                print('stop the path because number of active param reach n')
                return(BETA,SIGMA)
            
    pb.init = save_init
    pb.regpath = False
    return(BETA,SIGMA)








'''
Class of problem : we define a type, which will contain as keys, all the parameters we need for a given problem.
'''


class problem_Concomitant_Huber :
    
    def __init__(self,data,algo,rho):
        self.N = 500000
        
        if(len(data)==3): (A,C,y), self.dim = data, (data[0].shape[0],data[0].shape[1],data[1].shape[0])
        elif(len(data)==5):
            (A,C,sol), self.dim = generate_random(data), data[:3]
            self.sol,y = sol, A.dot(sol)+np.random.randn(data[0])*data[-1]
        self.matrix = (A,C,y)
        
        (m,d,k) = self.dim
        self.weights = np.ones(d)
        self.tol = 1e-6
         
          
        self.regpath = False
        self.name = algo + ' Concomitant Huber'
        self.type = algo          # type of algorithm used

        self.rho = rho
        self.mu  = 1.95
         
        self.c = (d/LA.norm(A,2))**2  # parameter for Concomitant problem : the matrix is scaled as c*A^2 
        self.gam = np.sqrt(d)
        sigmax = find_sigmax(y,rho)
        self.lambdamax = 2/sigmax*LA.norm((A.T).dot(h_prime(y,rho*sigmax)),np.infty)
        self.init = sigmax*np.ones(m),sigmax*np.ones(m),np.zeros(m), np.zeros(d), np.zeros(d)
           

        



'''
Functions used in the algorithms, modules needed : 
import numpy as np
import numpy.linalg as LA
from .../class_of_problem import problem
'''


# compute the prox of the function : f(b)= sum (wi * |bi| )
def prox(b,w,zeros): return(np.minimum(b+w,zeros)+np.maximum(b-w,zeros)) 

# Compute I - C^t (C.C^t)^-1 . C : the projection on Ker(C)
def proj_c(M,d):
    if (LA.matrix_rank(M)==0):  return(np.eye(d))
    return(np.eye(d)-LA.multi_dot([M.T,np.linalg.inv(M.dot(M.T) ),M]) )



def QQ(coef,A): return(coef*(A.T).dot(LA.inv(2*np.eye(A.shape[0])+coef*A.dot(A.T))),LA.inv(2*np.eye(A.shape[1])+coef*(A.T).dot(A)))    



# Compute the real positive root of a polynomial of degree 3 in the form : X^3 + a*X - b with Newton method and a warm start (for Comcomitant problem)
def calc_Newton(a,b,root):
    er = -b
    while (abs(er)>1e-6): 
        root= root-er/(3*root**2+a)
        er = root**3 + a*root-b
    return(root)



    
def prox_phi_2(sig,u,gamma,warm_start,rho):
    p,q, ws = np.zeros(len(u)), np.zeros(len(u)),  np.zeros(len(u))
    for i in range(len(u)):
        p[i],q[i],ws[i]=prox_phi_i(sig[i],u[i],2*gamma,warm_start[i],rho)
    return(p,q,ws)


def prox_phi_i(s,u,gamma,root,rho):
    if (u==0.): return(0,0,root)
    frac = gamma*rho/abs(u)
    term = s+gamma*(rho**2-1)/2
    bool1 = (frac>=1) 
    bool2 = (abs(u)**2<=gamma*(gamma-2*s))
    bool3 = (term<=0)
    bool4 = (abs(u)>=rho*s + gamma*rho*(1+rho**2)/2)
    
    
    if   (bool1  and bool2 ): return(0.,0.,root)
    elif (bool3 and not bool1): return(0.,u*(1-frac),root) 
    elif (not bool3 and bool4):  return(term,u*(1-frac),root)
        
    root = calc_Newton(2*s/gamma+1,2*abs(u)/gamma,root)
    return(s+gamma*(root**2-1)/2,u-gamma*root*np.sign(u),root)


# Compute the derivative of the huber function, particulary useful for the computing of lambdamax 
def h_prime(y,rho):
    m = len(y)
    lrho = rho*np.ones(m)
    return(np.maximum(lrho,-y)+ np.minimum(y-lrho,0))




def find_sigmax(y,rho):
    m,evol = len(y), True
    F = [True]*m
    if (rho > 1):
        while(evol):
            evol = False
            t = np.sqrt(m-(m-sum(F))*rho**2)/LA.norm(y[F])
            cste= rho/t
            for j in range(m):
                if (F[j] and y[j]>cste): F[j],evol= False , True
        return(1/t)
    else:
        print('rho too little ==> sigma is always 0')



def generate_random(dim):                # Function to generate random A, y, and C with given dimensions 
        (m,d,k,d_nonzero,sigma) = dim
        A, sol, sol_reduc= np.random.randn(m, d),np.zeros(d), np.random.rand(d_nonzero)
        if (k==0):
            C , list_i = np.zeros((1,d)), np.random.randint(d, size=d_nonzero)
            sol[list_i]=sol_reduc
        else:
            rank1,rank2 = 0,0
            while (rank1 !=k) :        
                C = np.random.randint(low=-1,high=2, size=(k, d))
                rank1 = LA.matrix_rank(C)
            while (rank2!=k):
                list_i = np.random.randint(d, size=d_nonzero)
                C_reduc = np.array([C.T[i] for i in list_i]).T
                rank2 = LA.matrix_rank(C_reduc)
            proj = proj_c(C_reduc,d_nonzero).dot(sol_reduc)
            sol[list_i]=proj
            return(A,C,sol)

