from CLasso.path_algorithm import solve_path
import numpy as np
import numpy.linalg as LA
    
'''    
Problem    :   min ||Ab - y||^2/sigma + 0.5 sigma + lambda ||b||1 with C.b= 0 and sigma > 0

Dimensions :   A : m*d  ;  y : m  ;  b : d   ; C : k*d

The first function compute a solution of a Lasso problem for a given lambda. The parameters are lam (lambda/lambdamax, \in [0,1]) and pb, which has to be a 'problem_LS type', which is defined bellow in order to contain all the important parameters of the problem. One can initialise it this way : pb = class_of_problem.problem(data=(A,C,y),type_of_algo). We solve the problem without normalizing anything. 
'''    



def algo_Concomitant(pb,lam, compute=True):
    
    pb_type = pb.type                          # ODE, 2prox
    (m,d,k),(A,C,y)  = pb.dim,pb.matrix
    lamb = lam * pb.lambdamax
    sigmax= LA.norm(y) * np.sqrt(2)

    #ODE
    # here we compute the path algo until our lambda, and just take the last beta
    if(pb_type == 'ODE'):       
                         
        (beta1,beta2), (s1,s2), (r1,r2) = solve_path((A,C,y),lam,concomitant = 'fix_lam')
        dr,ds = r1-r2,s1-s2
        teta  = root_2(LA.norm(dr)**2-ds**2 ,np.vdot(dr,r2)-s2*ds,LA.norm(r2)**2-s2**2)
        s     = s1    *teta + s2    *(1-teta)
        beta  = beta1 *teta + beta2 *(1-teta)
        return(beta,s*sigmax)
    

    if(compute): pb.compute_param()
    
    Anorm         = LA.norm(A,'fro')
    tol,regpath   = pb.tol * LA.norm(y)/Anorm,pb.regpath   # tolerance rescaled
    Proj          = proj_c(C,d)   # Proj = I - C^t . (C . C^t )^-1 . C
    AtA, Aty      = pb.AtA, pb.Aty     # Save some matrix products already computed in problem.compute_param() 
    gamma         = pb.gam / (pb.AtAnorm*lam)   # Normalize gamma 
    w,zerod       = lamb *gamma*pb.weights, np.zeros(d) # two vectors usefull to compute the prox of f(b)= sum(wi |bi|)
    mu, c, root   = pb.mu, pb.c, 0.
    Q1,Q2         = QQ(c,A) 
    QA,qy         = Q1.dot(A), Q1.dot(y)
    proj_sigm     = lambda x : max(x,0)
    xs,nu,o,xbar,x= pb.init

 
    #2prox
    if (pb_type == '2prox' ):
        for i in range(pb.N):
            nv_b, nv_s = x + Q1.dot(o) - QA.dot(x) - Q2.dot(x-xbar), (xs+nu)/2
            if (i>0 and LA.norm(b-nv_b)+LA.norm(s-nv_s)/Anorm <2*tol): 
                if (regpath):return(b,(xs,nu,o,xbar,x),s) 
                else :       return(b,s)
                        
            s,b = nv_s, nv_b
            Ab = A.dot(b)
            p1,p2,root = prox_phi_1(xs,2*Ab-o-y,gamma/c,root)
            sup = [ proj_sigm(nu)-s , p1-s , p2 + y - Ab , prox(2*b-xbar,w,zerod)-b, Proj.dot(2*b-x)-b]
            xs,nu,o,xbar,x = xs+mu*sup[0] ,  nu+mu*sup[1] ,  o+mu*sup[2] ,  xbar+mu*sup[3] ,  x+mu*sup[4]
            if (LA.norm(b)+LA.norm(s)>1e6): 
                print('DIVERGENCE')
                return(b,s)
        print('NO CONVERGENCE')
        return(b,'slow',s)

    
    print('none of the cases ! ')
    

    
    
    
    
    
    
    
    
'''
This function compute the the solution for a given path of lam : by calling the function 'algo' for each lambda with warm start, or wuth the method ODE, by computing the whole path thanks to the ODE that rules Beta and the subgradient s, and then to evaluate it in the given finite path.  
'''
    
def pathalgo_Concomitant(pb,path,n_active=False):
    n = pb.dim[0]
    BETA,SIGMA,tol = [],[],pb.tol

    
    
    if(pb.type=='ODE'):
        y=pb.matrix[2]
        sigmax= LA.norm(y) * np.sqrt(2)
        X,LAM,R = solve_path(pb.matrix,path[-1],concomitant = 'path',n_active=n_active)
        if (type(n_active)==int): return(X,[LA.norm(r)*sigmax for r in R],LAM)
        beta2,l2,r2,j = X[0],path[0]+0.1,-y/LA.norm(y),0
        for lam in path: 
            while (LA.norm(r2)<l2/lam): beta1,l1,r1,beta2,l2,r2,j = beta2,l2,r2,X[j],LAM[j],R[j],j+1
            s1,s2 = l1/lam,l2/lam
            dr,ds = r1-r2,s1-s2
            teta  = root_2(LA.norm(dr)**2-ds**2 ,np.vdot(dr,r2)-s2*ds,LA.norm(r2)**2-s2**2)
            SIGMA.append((s1*teta + s2*(1-teta))*sigmax)
            BETA.append(beta1 *teta + beta2 *(1-teta))
            
        return(BETA,SIGMA)

    
    save_init = pb.init   
    pb.regpath = True
    pb.compute_param()
    for lam in path:
        X = algo_Concomitant(pb,lam,compute=False)     
        BETA.append(X[0])
        SIGMA.append(X[-1])
        pb.init = X[1]
        if (type(n_active)==int) : n_act = n_active
        else : n_act = n
        if(sum([ (abs(X[0][i])>1e-2) for i in range(len(X[0])) ])>=n_act):
                pb.init = save_init
                BETA = BETA + [BETA[-1]]*(len(path)-len(BETA))
                pb.regpath = False
                return(BETA,SIGMA)
            
    pb.init = save_init
    pb.regpath = False
    return(BETA,SIGMA)








'''
Class of problem : we define a type, which will contain as keys, all the parameters we need for a given problem.
'''


class problem_Concomitant :
    
    def __init__(self,data,algo):
        self.N = 500000
        
        if(len(data)==3):
            (A,C,y) = data
            self.dim = (A.shape[0],A.shape[1],C.shape[0])
            e = self.dim[0]
            A,y = A/np.sqrt(2*e),y/np.sqrt(2*e)
            self.matrix = (A,C,y)
        
        elif(len(data)==5):
            (A,C,sol), self.dim = generate_random(data), data[:3]
            self.sol,y = sol, A.dot(sol)+np.random.randn(data[0])*data[-1]
            self.matrix = (A,C,y)

        
        (m,d,k) = self.dim
        self.weights = np.ones(d)
        
        self.tol = 1e-6
          
        self.regpath = False
        self.name = algo + ' Concomitant'
        self.type = algo          # type of algorithm used
        
        self.proj_sigm = lambda x : max(x,0)
        self.mu = 1.95

        self.gam        = np.sqrt(self.dim[1])
        self.Aty        = (A.T).dot(y)
        self.lambdamax  = np.sqrt(2)*LA.norm(self.Aty,np.infty)/LA.norm(y)
        sigmax          = np.sqrt(2)*LA.norm(y)
        self.init       = 1.,1.,np.zeros(m), np.zeros(d), np.zeros(d)    
            
            
            
    def compute_param(self):
        (A,C,y) = self.matrix
        m,d,k = self.dim
        self.c = (d/LA.norm(A,2))**2  # parameter for Concomitant problem : the matrix is scaled as c*A^2 
        
        self.AtA        =(A.T).dot(A)
        self.Cnorm      = LA.norm(C,2)**2
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


# Compute the real positive root of a polynomial of degree 3 in the form : X^3 + a*X - b with Newton method and a warm start (for Comcomitant problem)
def calc_Newton(a,b,root):
    er = -b
    while (abs(er)>1e-6): 
        root= root-er/(3*root**2+a)
        er = root**3 + a*root-b
    return(root)


def QQ(coef,A): return(coef*(A.T).dot(LA.inv(2*np.eye(A.shape[0])+coef*A.dot(A.T))),LA.inv(2*np.eye(A.shape[1])+coef*(A.T).dot(A)))    


# Return the cost function of some Beta for a given Lasso problem : L_LS = ||y-Ab||2^2  + lambda* ||b||1
def L_LS(pb,lam,sol): return(LA.norm( pb.matrix[0].dot(sol) - pb.matrix[2] )**2 + lam*pb.lambdamax * LA.norm(sol,1))


# Return the prox operator for the function phi = ||(y-Ab)||2^2 /sigma + 1/2 * sigma with the warm start on the computation of the root    
def prox_phi_1(sig,u,gamma,warm_start):
    l2u = LA.norm(u)
    Po0 = 4*gamma*sig + l2u**2
    if (Po0 < 2*gamma**2): return(0.,0.,0.)
    else : 
        root=calc_Newton((4*sig/gamma+6.),8*l2u/gamma,warm_start)
        return(sig+0.5*gamma*(0.5*root**2-1.),u * (1-gamma*root/l2u),root)


    
# Return the positive root in [0,1] of the polynomial aX^2 + 2bX + c if it exists
def root_2(a,b,c):
    root=-(np.sqrt(b**2-a*c)+b)/a
    if (root< 0 or root > 1) : return(0)
    return(root)








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



def courbes_gamma(lG,pb,lam):
    nG, C = len(lG), pb.matrix[1]
    T,F,D = np.zeros(nG), np.zeros(nG), np.zeros(nG)
    for j in range(nG) :
        pb.gam = lG[j]
        X = algo(pb,lam)
        if (X[1]==False): break
        T[j], F[j], D[j] = X[2], L_LS(pb,lam,X[0]), LA.norm(C.dot(X[0]))
    plt.figure(figsize=(14,7), dpi=80),plt.plot(lG,T, label = pb.name)
    plt.xlabel("gamma"),plt.ylabel("Running time"), plt.legend(), plt.show()
    
    plt.figure(figsize=(14,7), dpi=80),plt.plot(lG,F, label = pb.name)
    plt.xlabel("gamma"),plt.ylabel("Value of minimizing f"), plt.legend(), plt.show()    
    
    plt.figure(figsize=(14,7), dpi=80),plt.plot(lG,D, label = pb.name)
    plt.xlabel("gamma"),plt.ylabel("Violation of constraints"), plt.legend(), plt.show() 
