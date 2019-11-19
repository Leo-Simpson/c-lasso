import numpy as np
import numpy.linalg as LA
import scipy.io as sio


import Functions.functions_4algos as algoF


def generate_random(dim):                # Function to generate random A, y, and C
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
            proj = sol_reduc - (C_reduc.T).dot(algoF.proj_c(C_reduc).dot(sol_reduc))
            sol[list_i]=proj
            return(A,C,sol)



class problem :
    
    def __init__(self,data,ty,normalise=True):
        self.N = 500000
        
        if(len(data)==3):self.matrix, self.dim = data, (data[0].shape[0],data[0].shape[1],data[1].shape[0])
        
        elif(len(data)==5):
            (A,C,sol), self.dim = generate_random(data), data[:3]
            self.sol,y = sol, A.dot(sol)+np.random.randn(data[0])*data[-1]
            self.matrix = (A,C,y)
        
        (m,d,k) = self.dim
        self.weights = np.ones(d)
        self.init = 1.,1.,np.zeros(m), np.zeros(d), np.zeros(d), np.zeros(k)
        self.tol = 1e-7
         
        
        algo = ty
        self.regpath = False
        self.name = ty
        self.type = algo          # type of algorithm used
        
        self.prox_phi = algoF.prox_phi_1
        self.mu = 1.95
        
        
        
        self.r = 1             # parameter for Huber problem : LS + lambda |b|_1 + lambda/r |o|_1
        
        self.gam = 1.
        self.tau = 0.5         # equation for the convergence of Noproj and LS algorithms : 2.gam + tau < 1
        if (ty=='LSG' or ty == 'Concomitant'): self.gam = self.dim[1]**(1.5)
       
        if normalise: self.normalize()
        else: self.My, self.lnA,self.lny = 0., [1.]*d, 1.
            
        self.c = (d/LA.norm(self.matrix[0],2))**2  # parameter for Concomitant problem : the matrix is scaled as c*A^2 
        self.compute_param()
        
            
    def compute_param(self):
        (A,C,y) = self.matrix
        m,d,k = self.dim
        coef = self.c
        Q1 = coef*(A.T).dot(LA.inv(2*np.eye(A.shape[0])+coef*A.dot(A.T)))
        Q2 = LA.inv(2*np.eye(A.shape[1])+coef*(A.T).dot(A))
        self.Q = (Q1, Q2)
        if (LA.matrix_rank(C)==0):  self.projC = C
        else:     self.projC = np.linalg.inv(C.dot(C.T) ).dot( C )
        self.AtA =(A.T).dot(A)
        self.Aty = (A.T).dot(y)
        self.Cnorm  = LA.norm(C,2)**2
        self.tauN= self.tau/self.Cnorm
        self.lambdamax1 = 2*LA.norm((A.T).dot(y),np.infty)
        self.lambdamax = self.lambdamax1
        self.AtAnorm = LA.norm(self.AtA,2)
        
        
        
    def normalize(self):
        (A,C,y),(m,d,k) = self.matrix, self.dim
        self.My = sum(y)/m
        self.lnA = [LA.norm(A[:,j]) for j in range(d)]
        self.lny = LA.norm(y-self.My*np.ones(m))
        la,ly,my = self.lnA, self.lny , self.My
        An = np.array([[A[i,j]/(la[j])  for j in range(d)] for i in range(m)])
        yn = np.array([ (y[j]-my)/ly for j in range(m) ])
        Cn = np.array([[C[i,j]*ly/la[j] for j in range(d)] for i in range(k)])
        self.matrix = (An,Cn,yn)
        
    def denormalize(self):
        (A,C,y),(m,d,k),la,ly,my = self.matrix, self.dim, self.lnA, self.lny , self.My
        An = np.array([[A[i,j]*la[j]  for j in range(d)] for i in range(m)])
        Cn = np.array([[C[i,j]*la[j]/ly  for j in range(d)] for i in range(k)])
        yn = np.array([ (y[j]+my)*ly for j in range(m) ])
        self.matrix = (An,Cn,yn)  
        
        
    def change_type(self,typ):
        self.type,self.name = typ,typ
        self.gam , self.tauN  = .5 , .1/self.Cnorm
        if (typ=='LSG' or typ == 'Concomitant'): self.gam = np.sqrt(self.dim[1])
        if (typ == 'Concomitant'): self.lambdamax = self.lambdamax1 / np.sqrt(2)

            
            
            
            
            
            
            