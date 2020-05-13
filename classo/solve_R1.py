from .path_alg import solve_path
import numpy as np
import numpy.linalg as LA
    
'''    
Problem    :   min ||Ab - y||^2 + lambda ||b||1 with C.b= 0

Dimensions :   A : m*d  ;  y : m  ;  b : d   ; C : k*d

The first function compute a solution of a Lasso problem for a given lambda. 
The parameters are lam (lambda/lambdamax, in [0,1]) and pb, which has to be a 'problem_LS type'
 which is defined bellow in order to contain all the important parameters of the problem. 
'''    



def Classo_R1(pb,lam):
    pb_type = pb.type  # can be 'Path-Alg', 'P-PDS' , 'PF-PDS' or 'DR'

    # ODE
    # here we compute the path algo until our lambda, and just take the last beta
    if(pb_type == 'Path-Alg'):
        BETA = solve_path(pb.matrix, lam, False, 0, 'LS')[0]
        return(BETA[-1])
    
    (m,d,k),(A,C,y)  = pb.dim,pb.matrix            
    lamb = lam * pb.lambdamax
    tol = pb.tol * LA.norm(y)/LA.norm(A,'fro')  # tolerance rescaled
    regpath = pb.regpath 

    #cvx
    # call to the cvx function of minimization
    if (pb_type == 'cvx'):
        import cvxpy as cp
        lamb = lam*2*LA.norm(A.T.dot(y),np.infty)
        x = cp.Variable(d)
        objective, constraints = cp.Minimize(cp.sum_squares(A*x-y)+ lamb*cp.norm(x, 1)), [C*x == 0]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(warm_start=regpath,eps_abs= tol)
        if (regpath): return(x.value,True) 
        return(x.value)

    if(not regpath): pb.compute_param()   # this is a way to compute costful matrices computation like A^tA only once when we do pathcomputation with warm starts.
    Proj,AtA, Aty = proj_c(C,d), pb.AtA, pb.Aty     # Save some matrix products already computed in problem.compute_param()
    gamma, tau    = pb.gam / (2*pb.AtAnorm),    pb.tauN
    w,zerod       = lamb *gamma*pb.weights, np.zeros(d) # two vectors usefull to compute the prox of f(b)= sum(wi |bi|)


    # NO PROJ
    
    if (pb_type == 'PF-PDS'):     # y1 --> S ; p1 --> p . ; p2 --> y2
        (x,v) = pb.init
        for i in range(pb.N):    
            
            S = x - gamma*(AtA.dot(x)-Aty)*2 - (C.T).dot(v)
            p = prox(S,w,zerod)
            
            y2 = v + tau*C.dot(x)
            v  = v + tau*C.dot(p)
            
            eps = p - gamma*(AtA.dot(p)-Aty)*2 - C.T.dot(y2)-S
            x = x + eps
            
            if (i>0 and LA.norm(eps)<tol): 
                if (regpath): return(x,(x,v)) 
                else :        return(x) 
            
            if (LA.norm(x)+LA.norm(p)+LA.norm(v)>1e6): 
                print('DIVERGENCE')
                return(x)
        print('NO CONVERGENCE')
        return(x)
    

    #FORARD BACKWARD
    
    if (pb_type=='P-PDS'):
        xbar,x,v = pb.init
        for i in range(pb.N):                   
            grad = (AtA.dot(x)-Aty)
            v = v + tau*C.dot(xbar)            
            s = x - 2*gamma*grad - (C.T).dot(v) 
            p = prox(s,w,zerod)              
            nw_x = Proj.dot(p)
            
            eps = nw_x - x                       
            xbar= p + eps                        
            
            
            if (i%10==2 and LA.norm(eps)<tol):      # 0.6
                if (regpath): return(x,(xbar,x,v)) 
                else :        return(x) 
            x= nw_x 
            if (LA.norm(x)>1e10): 
                print('DIVERGENCE')
                return(x)
        print('NO CONVERGENCE')
        return(x)

    gamma             = gamma/(2*lam)
    w                 = w /(2*lam)
    mu,ls, c, root    = pb.mu,[], pb.c, 0.
     # 2 PROX  
    if (pb_type=='DR'):
        
        Q1,Q2  = QQ(2*gamma/(mu-1),A)
        QA,qy = Q1.dot(A), Q1.dot(y)
        
        qy_mult = qy*(mu-1)
        
        b,xbar,x = pb.init
        for i in range(pb.N):
            xbar= xbar + mu*(prox(2*b-xbar,w,zerod)-b)
            x   = x    + mu*(Proj.dot(2*b-x)       -b)

            nv_b = (2-mu)*b
            nv_b = nv_b + qy_mult + Q2.dot(x+xbar- 2*nv_b)
            if (i%2==1 and LA.norm(b-nv_b)<tol): 
                if (regpath):
                    return(b,(b,xbar,x))
                else :       return(b)  
    
            b = nv_b

        print('NO CONVERGENCE')
        return(b)  
 

    
'''
This function compute the the solution for a given path of lam : by calling the function 'algo' for each lambda with warm start, or with the method ODE, by computing the whole path thanks to the ODE that rules Beta and the subgradient s, and then to evaluate it in the given finite path.
'''
    
def pathlasso_R1(pb,path,n_active=False,return_sp_path=False):
    n = pb.dim[0]
    BETA,tol = [],pb.tol
    if(pb.type == 'Path-Alg'):
        beta,sp_path = solve_path(pb.matrix,path[-1],n_active, 0, 'LS')
        if (return_sp_path): return(beta,sp_path) # in the method ODE, we only compute the solution for breaking points. We can stop here if return_sp_path=True
        else : # else, we do a little manipulation to interpolated the value of beta between those points, as we know beta is affine between those breaking points.
            sp_path.append(path[-1]),beta.append(beta[-1])
            i=0
            for lam in path:
                while (lam<sp_path[i+1]): i+=1
                teta = (sp_path[i]-lam)/(sp_path[i]-sp_path[i+1])
                BETA.append(beta[i]*(1-teta)+beta[i+1]*teta)
            return(BETA)
    
    
    # Now we are in the case where we have to do warm starts.
    save_init = pb.init   
    pb.regpath = True
    pb.compute_param()
    for lam in path:
        X = Classo_R1(pb,lam)
        BETA.append(X[0])
        pb.init = X[1]
        if (type(n_active)==int) : n_act = n_active
        else : n_act = n
        if(sum([ (abs(X[0][i])>1e-4) for i in range(len(X[0])) ])>=n_act):
                pb.init = save_init
                BETA = BETA + [BETA[-1]]*(len(path)-len(BETA))
                pb.regpath = False
                return(BETA)
            
    pb.init = save_init
    pb.regpath = False
    return(BETA)








'''
Class of problem : we define a type, which will contain as attributes all the parameters we need for a given problem.
'''


class problem_R1 :
    
    def __init__(self,data,algo):
        self.N = 500000
        
        if(len(data)==3):self.matrix, self.dim = data, (data[0].shape[0],data[0].shape[1],data[1].shape[0])
        
        elif(len(data)==5):
            (A,C,sol), self.dim = generate_random(data), data[:3]
            self.sol,y = sol, A.dot(sol)+np.random.randn(data[0])*data[-1]
            self.matrix = (A,C,y)
        
        (m,d,k) = self.dim
        
        if(algo=='P-PDS') : self.init = np.zeros(d), np.zeros(d), np.zeros(k)
        elif (algo=='PF-PDS')         : self.init = np.zeros(d), np.zeros(k)
        else                        : self.init = np.zeros(d), np.zeros(d), np.zeros(d)
        self.tol = 1e-6 
         
        self.weights = np.ones(d)  
        self.regpath = False
        self.name = algo + ' LS'
        self.type = algo          # type of algorithm used
        self.mu = 1.95
        self.Aty= (self.matrix[0].T).dot(self.matrix[2])
        self.lambdamax = 2*LA.norm(self.Aty,np.infty)
        self.gam = 1.
        self.tau = 0.5         # equation for the convergence of 'PF-PDS' and LS algorithms : gam + tau < 1
        if (algo =='DR'): self.gam = self.dim[1]


    #this is a method of the class pb that is used to computed the expensive multiplications only once. (espacially usefull for warm start. )

    def compute_param(self):
        (A,C,y) = self.matrix
        m,d,k = self.dim
        self.c = (d/LA.norm(A,2))**2  # parameter for Concomitant problem : the matrix is scaled as c*A^2 

        self.AtA        =(A.T).dot(A)
        self.Cnorm      = LA.norm(C,2)**2
        self.tauN       = self.tau/self.Cnorm
        self.AtAnorm    = LA.norm(self.AtA,2)

        
        
        
        



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

