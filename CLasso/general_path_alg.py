N=10000
import numpy as np
import numpy.linalg as LA


class parameters_for_update:
    '''Object that has parameters needed at each breaking point during path algorithm

    Attributes :
        number_act  (int) : number of active parameter
        idr         : saves the independant rows of the matrix C resctricted to the actives parameters
        Xt          :  inverse of M
        activity    :  list of boolean, activity[i] is True when variable i is active
        beta        : current solution beta
        s           : current subgradient
        lam         : current lam
        lambdamax   : lambdamax
        M           : matrix to invert
        y           : output
        r           : residual
        F           : F is the set where r<1 and if huber, then it is the set where rho<r<1
        rho         : only use when doing huber path algo


    '''
    def __init__(self,matrices,lamin, rho):

        (self.A, self.C, self.y) = matrices
        self.lamin = lamin
        self.rho = rho
        n, d, k = len(self.A), len(self.A[0]), len(self.C)
        self. number_act = 0
        self.idr         = [False] * k
        self.activity    = [False] * d
        self.beta        = np.zeros(d)
        s                = 2*self.A.T.dot(h_prime(self.y,rho))
        self.lambdamax   = LA.norm(s, np.inf)
        self.s           = s/self.lambdamax
        self.lam         = 1.
        self.r           = -self.y
        self.F           = [True] * n
        if (rho!=0):
            for j in range(n):
                if( abs(self.y[j]) > self.rho ): self.F[j] = False
        AtA = self.A[self.F].T.dot(self.A[self.F])
        for i in range(d):
            if (self.s[i] == 1. or self.s[i] == -1.):
                self.activity[i] = True
                self.number_act += 1
                if (k > 0):
                    to_ad = next_idr1(self.idr, self.C[:, self.activity])
                    if (type(to_ad) == int): self.idr[to_ad] = True


        if(k==0):  self.M = 2 * AtA
        else    :  self.M  = np.concatenate((np.concatenate((2 * AtA, self.C.T), axis=1), np.concatenate((self.C, np.zeros((k, k))), axis=1)),axis=0)

        self.Xt          = LA.inv(self.M[self.activity + self.idr, :][:, self.activity + self.idr])




def solve_path(matrices,lamin,up, n_active,rho):
    param = parameters_for_update(matrices,lamin,rho)
    BETA, LAM = [param.beta] , [param.lam]
    for i in range(N) :
        up(param)
        BETA.append(param.beta), LAM.append(param.lam)
        if ((type(n_active)==int and param.number_act>= n_active) or param.lam == lamin): return(BETA,LAM)
            
    print('no conv')
    return(BETA,LAM)



#Upddate a list of constraints which are independant if we restrict the matrix C to the acrive set (C_A has to have independant rows)

#When we ad an active parameter
def next_idr1(liste,mat):
    if(sum(liste)==len(mat)): return(False)
    if (sum(liste)==0):
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if not (mat[i,j]==0): return(i)
        return(False)
    Q = LA.qr(mat[liste,:].T)[0]
    for j in range(len(mat)):
        if (not liste[j]) and (  LA.norm(mat[j]-LA.multi_dot([Q,Q.T,mat[j]]))>1e-10 ): return(j)
    return(False) 



# Fonction to interpolate the solution path between the breaking points
def pathalgo_general(matrix,path,up,n_active=False,rho=0):
    BETA, i = [], 0
    X,sp_path = solve_path(matrix,path[-1],up,n_active,rho)
    sp_path.append(path[-1]),X.append(X[-1])
    for lam in path:
        while (lam<sp_path[i+1]): i+=1
        teta = (sp_path[i]-lam)/(sp_path[i]-sp_path[i+1])
        BETA.append(X[i]*(1-teta)+X[i+1]*teta)
    return(BETA)


# Compute the derivative of the huber function, particulary useful for the computing of lambdamax
def h_prime(y,rho):
    if (rho==0): return(y)
    m = len(y)
    lrho = rho*np.ones(m)
    return(np.maximum(lrho,-y)+ np.minimum(y-lrho,0))
    
