from CLasso.path_algorithm_huber_classification import solve_huber_cl_path
import numpy as np
import numpy.linalg as LA
from CLasso.solve_LS import problem_LS, algo_LS
    
'''    
Problem    :   min h_rho(Ab - y) + lambda ||b||1 with C.b= 0 <=>   min ||Ab - y - r*o||^2 + lambda ||b,o||1 with C.b= 0, o in R^m
                                                                                        r = lambda / 2rho
Dimensions :   A : m*d  ;  y : m  ;  b : d   ; C : k*d

The first function compute a solution of a Lasso problem for a given lambda. The parameters are lam (lambda/lambdamax, \in [0,1]) and pb, which has to be a 'problem_LS type', which is defined bellow in order to contain all the important parameters of the problem. One can initialise it this way : pb = class_of_problem.problem(data=(A,C,y),type_of_algo). We solve the problem without normalizing anything. 
'''    


def algo_Huber_Cl(pb,lam, compute=True):
    
    pb_type = pb.type   # ODE, FB, cvx
    
    
    (m,d,k),(A,C,y)  = pb.dim,pb.matrix        
    lamb, rho = lam * pb.lambdamax,  pb.rho
    
    
    #ODE
    # here we compute the path algo until our lambda, and just take the last beta
    
    if(pb_type == 'ODE'):                         
        beta = solve_huber_path((A,C,y), lam,rho)[0]
        return(beta[-1])
    
    tol = pb.tol * LA.norm(y)/LA.norm(A,'fro')  # tolerance rescaled
    regpath = pb.regpath
    
    #cvx 
    # call to the cvx function of minimization
    if (pb_type == 'cvx'):
        import cvxpy as cp
        x = cp.Variable(d)
        objective, constraints = cp.Minimize(cp.sum(cp.huber(A*x - y, rho))+lamb*cp.norm(x, 1)), [C*x == 0]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(warm_start=regpath,eps_abs= tol)
        if (regpath): return(x.value,True) 
        return(x.value)
     
    if(compute): pb.compute_param()
    tau,r,Proj, AtA, Aty = pb.tauN, lamb/(2*rho),  proj_c(C,d), pb.AtA, pb.Aty  
    gamma                 = pb.gam / (2*(pb.AtAnorm+r**2))   
    t                     = lamb * gamma
    w,tm,zerom,zerod      = t*pb.weights,t*np.ones(m), np.zeros(m), np.zeros(d) 
    o,xbar,x,v = pb.init
    # vectors usefull to compute the prox of f(b)= sum(wi |bi|)
    
    #FORWARD BACKWARD
    if (pb_type=='FB'):
        
        for i in range(pb.N):                        
            grad = (AtA.dot(x)-Aty)                  
            v = v + tau*C.dot(xbar)             
            S = x - 2*gamma*grad - 2*gamma*r*(A.T).dot(o) - (C.T).dot(v) 
            o = prox(o*(1-2*gamma*r**2) + 2*gamma*r*(y-A.dot(x)),tm,zerom)
            p = prox(S,w,zerod)             
            nw_x = Proj.dot(p)                   
            eps = nw_x - x                      
            xbar= p + eps                     

            if (i%10==2 and LA.norm(eps)<tol):      # 0.6
                if (regpath): return(x,(o,xbar,x,v)) 
                else :        return(x) 
            x= nw_x 
            if (LA.norm(x)>1e10): 
                print('DIVERGENCE')
                return(x)
        print('NO CONVERGENCE')
        return(x)
     
        
    # NO PROJ
    
    if (pb_type == 'Noproj'):     # y1 --> S ; p1 --> p . ; p2 --> y2
        for i in range(pb.N):    
            grad = (AtA.dot(x)-Aty)
            
            S1 = x - 2*gamma*grad - 2*gamma*r*(A.T).dot(o) - (C.T).dot(v)
            S2 = o*(1-2*gamma*r**2) + 2*gamma*r*(y-A.dot(x))
            
            p1 = prox(S1,w,zerod)
            p2 = prox(S2,tm,zerom)
            
            v   = v + tau*C.dot(p1)
            v2  = v + tau*C.dot(x)
            
            eps1 = p1 + 2*gamma*(Aty-AtA.dot(p1)-r*A.T.dot(o)) - C.T.dot(v2)-S1
            eps2 = p2 + 2*r*gamma*(y-r*p2-A.dot(x)) -S2
            
            x = x + eps1
            o = o + eps2
            
            if (i>0 and LA.norm(eps1)+LA.norm(eps2)<tol): 
                if (regpath): return(x,(o,xbar,x,v)) 
                else :        return(x) 
            
            if (LA.norm(x)+LA.norm(o)+LA.norm(v)>1e6): 
                print('DIVERGENCE')
                return(x)
        print('NO CONVERGENCE')
        return(x)
    
    
    
    
    # 2 prox :
    
    if (pb_type == '2prox'):
        Ahuber = np.concatenate((A,r*np.eye(len(A))),axis = 1 )
        Chuber = np.concatenate((C,np.zeros((len(C),len(y)))),axis = 1 )
        matrices_huber = (Ahuber,Chuber,y)
        prob = problem_LS(matrices_huber,'2prox')
        prob.regpath = regpath
        if (len(pb.init)==3): prob.init = pb.init
        if not (regpath): return(algo_LS(prob,lamb/prob.lambdamax)[:d])
        x , warm_start = algo_LS(prob,lamb/prob.lambdamax)
        return(x[:d],warm_start)
    
    

    
'''
This function compute the the solution for a given path of lam : by calling the function 'algo' for each lambda with warm start, or wuth the method ODE, by computing the whole path thanks to the ODE that rules Beta and the subgradient s, and then to evaluate it in the given finite path.  
'''
    
    
def pathalgo_Huber_Cl(pb,path,n_active=False):
    n = pb.dim[0]
    BETA,tol = [],pb.tol
    if(pb.type == 'ODE'):
        X,sp_path = solve_huber_path(pb.matrix,path[-1],pb.rho,n_active)
        i=0
        sp_path.append(path[-1]),X.append(X[-1])
        for lam in path:
            while (lam<sp_path[i+1]): i+=1
            teta = (sp_path[i]-lam)/(sp_path[i]-sp_path[i+1])
            BETA.append(X[i]*(1-teta)+X[i+1]*teta)
        return(BETA)
    
    save_init = pb.init   
    pb.regpath = True
    pb.compute_param()
    if (type(n_active)==int) : n_act = n_active
    else : n_act = n
    for lam in path:
        X = algo_Huber(pb,lam,compute=False)  
        BETA.append(X[0])
        pb.init = X[1]
        if(sum([ (abs(X[0][i])>1e-2) for i in range(len(X[0])) ])>=n_act):
                pb.init = save_init
                BETA = BETA + [BETA[-1]]*(len(path)-len(BETA))
                pb.regpath = False
                return(BETA)
            
    pb.init = save_init
    pb.regpath = False
    return(BETA)








'''
Class of problem : we define a type, which will contain as keys, all the parameters we need for a given problem.
'''


class problem_Huber_Cl :
    
    def __init__(self,data,algo,rho_to_normalise):
        self.N = 500000
        
        if(len(data)==3):self.matrix, self.dim = data, (data[0].shape[0],data[0].shape[1],data[1].shape[0])
        
        elif(len(data)==5):
            (A,C,sol), self.dim = generate_random(data), data[:3]
            self.sol,y = sol, A.dot(sol)+np.random.randn(data[0])*data[-1]
            self.matrix = (A,C,y)
        
        (m,d,k) = self.dim
        rho = rho_to_normalise * LA.norm(self.matrix[2],np.infty)/np.sqrt(m)
        self.weights = np.ones(d)
        self.init = np.zeros(m), np.zeros(d), np.zeros(d), np.zeros(k)
        self.tol = 1e-6
        self.regpath = False
        self.name = algo + ' Huber'
        self.type = algo        
        self.rho = rho
        self.gam = 1.
        self.tau = 0.5         # equation for the convergence of Noproj and LS algorithms : gam + tau < 1
        self.lambdamax = 2*LA.norm(self.matrix[0].T.dot(h_prime(self.matrix[2],rho)),np.infty)
            
        
           
            
            
            
            
    def compute_param(self):
        (A,C,y) = self.matrix
        m,d,k = self.dim
        self.c = (d/LA.norm(A,2))**2  # parameter for Concomitant problem : the matrix is scaled as c*A^2 
        

        self.AtA        =(A.T).dot(A)
        self.Aty        = (A.T).dot(y)
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

# Compute the derivative of the huber function, particulary useful for the computing of lambdamax 
def h_prime(y,rho):
    m = len(y)
    lrho = rho*np.ones(m)
    return(np.maximum(lrho,-y)+ np.minimum(y-lrho,0))
        






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

