from .path_alg import solve_path
import numpy as np
import numpy.linalg as LA
from .solve_R1 import problem_R1, Classo_R1
    
'''    
Problem    :   min h_rho(Ab - y) + lambda ||b||1 with C.b= 0 <=>   min ||Ab - y - r*o||^2 + lambda ||b,o||1 with C.b= 0, o in R^m
                                                                                        r = lambda / 2rho
Dimensions :   A : m*d  ;  y : m  ;  b : d   ; C : k*d

The first function compute a solution of a Lasso problem for a given lambda. The parameters are lam (lambda/lambdamax, \in [0,1]) and pb, which has to be a 'problem_LS type', which is defined bellow in order to contain all the important parameters of the problem. One can initialise it this way : pb = class_of_problem.problem(data=(A,C,y),type_of_algo). We solve the problem without normalizing anything. 
'''    


def Classo_R2(pb,lam, compute=True):
    
    pb_type = pb.type    # can be 'Path-Alg', 'P-PDS' , 'PF-PDS' or 'DR'
    
    
    (m,d,k),(A,C,y)  = pb.dim,pb.matrix        
    lamb, rho = lam * pb.lambdamax,  pb.rho

    if (lam == 0.): pb_type, compute = 'DR', True  #here we simply refer to Classo_R1 that is called line 42. 

    
    
    
    #ODE
    # here we compute the path algo until our lambda, and just take the last beta
    
    if(pb_type == 'Path-Alg'):
        beta = solve_path((A,C,y), lam,False,rho,'huber')[0]
        return(beta[-1])

    # 2 prox :
    regpath = pb.regpath
    r = lamb / (2 * rho)
    if (pb_type == 'DR'):
        if compute : 
            pb.init_R1(r=r)
            return Classo_R1(pb.prob_R1, lamb / pb.prob_R1.lambdamax)[:d]
        else : 
            pb.add_r(r=r)
            if len(pb.init)==3:pb.prob_R1.init = pb.init
            x, warm_start = Classo_R1(pb.prob_R1, lamb / pb.prob_R1.lambdamax)
            return (x[:d], warm_start)

    tol = pb.tol * LA.norm(y)/LA.norm(A,'fro')  # tolerance rescaled
    
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
    tau,Proj, AtA, Aty = pb.tauN,  proj_c(C,d), pb.AtA, pb.Aty
    gamma                 = pb.gam / (2*(pb.AtAnorm+r**2))   
    t                     = lamb * gamma
    w,tm,zerom,zerod      = t*pb.weights,t*np.ones(m), np.zeros(m), np.zeros(d) 
    o,xbar,x,v = pb.init
    # vectors usefull to compute the prox of f(b)= sum(wi |bi|)
    
    #FORWARD BACKWARD
    if (pb_type=='P-PDS'):
        
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
                raise ValueError("The algorithm of P-PDS diverges")

        raise ValueError("The algorithm of P-PDS did not converge after %i iterations " %pb.N)
     
        
    # NO PROJ
    
    if (pb_type == 'PF-PDS'):     # y1 --> S ; p1 --> p . ; p2 --> y2
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
                raise ValueError("The algorithm of PF-PDS diverges")

        raise ValueError("The algorithm of PF-PDS did not converge after %i iterations " %pb.N)
    
    
    

    
'''
This function compute the the solution for a given path of lam : by calling the function 'algo' for each lambda with warm start, or wuth the method ODE, by computing the whole path thanks to the ODE that rules Beta and the subgradient s, and then to evaluate it in the given finite path.  
'''
    
    
def pathlasso_R2(pb,path,n_active=False):
    n = pb.dim[0]
    BETA,tol = [],pb.tol
    if(pb.type == 'Path-Alg'):
        X,sp_path = solve_path(pb.matrix,path[-1],n_active,pb.rho, 'huber')
        # we do a little manipulation to interpolated the value of beta between breaking points, as we know beta is affine between those those points.
        i=0
        sp_path.append(path[-1]),X.append(X[-1])
        for lam in path:
            while (lam<sp_path[i+1]): i+=1
            teta = (sp_path[i]-lam)/(sp_path[i]-sp_path[i+1])
            BETA.append(X[i]*(1-teta)+X[i+1]*teta)
        return(BETA)

    # Now we are in the case where we have to do warm starts.
    save_init = pb.init   
    pb.regpath = True
    pb.compute_param()
    pb.init_R1()
    if type(n_active)==int and n_active > 0: n_act = n_active
    else : n_act = n
    for lam in path:
        X = Classo_R2(pb,lam,compute=False)
        BETA.append(X[0])
        pb.init = X[1]
        if sum([ (abs(X[0][i])>1e-1) for i in range(len(X[0])) ])>=n_act or type(X[1])==str :
                pb.init = save_init
                BETA.extend( [BETA[-1]]*(len(path)-len(BETA)) )
                pb.regpath = False
                return(BETA)
            
    pb.init = save_init
    pb.regpath = False
    return(BETA)








'''
Class of problem : we define a type, which will contain as keys, all the parameters we need for a given problem.
'''


class problem_R2 :
    
    def __init__(self,data,algo,rho):
        self.N = 500000
        
        self.matrix, self.dim = data, (data[0].shape[0],data[0].shape[1],data[1].shape[0])
        
        (m,d,k) = self.dim
        self.weights = np.ones(d)
        self.init = np.zeros(m), np.zeros(d), np.zeros(d), np.zeros(k)
        self.tol = 1e-4
        self.regpath = False
        self.name = algo + ' Huber'
        self.type = algo        # type of algo used
        self.rho = rho
        self.gam = 1.
        self.tau = 0.5         # equation for the convergence of Noproj and LS algorithms : gam + tau < 1
        self.lambdamax = 2*LA.norm(self.matrix[0].T.dot(h_prime(self.matrix[2],rho)),np.infty)



    '''
    this is a method of the class pb that is used to computed the expensive multiplications only once. (espacially usefull for warm start. )
    '''
    def compute_param(self):
        (A,C,y) = self.matrix
        m,d,k = self.dim
        self.c = (d/LA.norm(A,2))**2  # parameter for Concomitant problem : the matrix is scaled as c*A^2

        self.AtA        =(A.T).dot(A)
        self.Aty        = (A.T).dot(y)
        self.Cnorm      = LA.norm(C,2)**2
        self.tauN       = self.tau/self.Cnorm
        self.AtAnorm    = LA.norm(self.AtA,2)


    def init_R1(self,r=0.):
        (m,d,k) = self.dim
        Ahuber = np.append(self.matrix[0], r * np.eye(m), 1)
        Chuber = np.append(self.matrix[1], np.zeros((k, m)), 1)
        matrices_huber = (Ahuber, Chuber, self.matrix[2])
        prob = problem_R1(matrices_huber, 'DR')
        prob.regpath = self.regpath
        prob.compute_param()

        self.prob_R1 = prob

    def add_r(self,r):
        prob=self.prob_R1
        d = prob.dim[1]-prob.dim[0]
        np.fill_diagonal(prob.matrix[0][d:] , r ) 
        np.fill_diagonal(prob.AtA[d:,d:], r**2   )
        prob.AtA[d:,:d] = prob.matrix[0][:,:d]*r
        prob.AtA[:d,d:] = prob.matrix[0][:,:d].T*r

        prob.Aty = np.append(prob.Aty,prob.matrix[2]*r)
        prob.lambdamax = 2*LA.norm(prob.Aty,np.infty)




        


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


