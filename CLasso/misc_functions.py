colo = ['red','orange','g','b','pink','y','c','m','purple',
        'yellowgreen','silver','coral','plum','lime',
        'hotpink','palegreen', 'tan', 'firebrick','darksalmon',
        'sienna', 'sandybrown','olive', 'cadetblue','lawngreen',
        'palevioletred','papayawhip','turquoise', 'teal',
        'khaki','peru','indianred','brown', 'slategrey']
colo = colo*100
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from scipy.special import erfinv

def random_data(n,d,d_nonzero,k,sigma,zerosum=False,seed=False, classification = False):
    ''' Generation of random matrices as data such that y = X.sol + sigma. noise

    The data X is generated as a normal matrix
    The vector sol is generated randomly with a random support of size d_nonzero,
    and componants are projected random intergers between -10  and 10 on the kernel of C restricted to the support
    The vector y is then generated with X.dot(sol)+ sigma*noise , with noise a normal vector


    Args:
        n (int): Number of sample, dimension of y
        d (int): Number of variables, dimension of sol
        d_nonzero (int): Number of non null componant of sol
        k (int) : Number of constraints, number of rows of C
        sigma (float) : size of standard error
        zerosum (bool, optional) : If True, then C is the all-one matrix with 1 row, independently of k
        seed (bool or int, optional) : Seed for random values, for an equal seed, the result will be the same. If set to False: pseudo-random vectors
        classification (bool, optional) : if True, then it returns sign(y) instead of y

    Returns:
        tuple : tuple of three ndarray that corresponds to the data :  (X,C,y)
        ndarray : array corresponding to sol which is the real solution of the problem y = Xbeta + noise s.t. beta sparse and Cbeta = 0
    '''
    if (type(seed) == int): np.random.seed(seed)
    else : np.random.seed()
    X= np.random.randn(n,d)
    if (zerosum): C,k = np.ones((1,d)),1
    else :
        if (k==0):
            sol, sol_reduc,list_i = np.zeros(d), np.random.randint(-10,11,d_nonzero),np.random.randint(d, size=d_nonzero)
            sol[list_i]=sol_reduc
            y = X.dot(sol)+np.random.randn(m)*sigma
            return((X,np.zeros((0,d)),y),sol)
        while True :        
            C = np.random.randint(low=-1,high=1, size=(k, d))
            if (LA.matrix_rank(C)==k): break
    while True:
        sol, sol_reduc = np.zeros(d), np.random.randint(-10,11,d_nonzero)
        list_i = np.random.randint(d, size=d_nonzero)
        C_reduc = np.array([C.T[i] for i in list_i]).T
        if (LA.matrix_rank(C_reduc)<k): continue
        proj = proj_c(C_reduc,d_nonzero).dot(sol_reduc)
        for i in range(len(list_i)):
            sol[list_i[i]]=proj[i]
        break
    y = X.dot(sol)+np.random.randn(n)*sigma
    if classification : y = np.sign(y)
    return((X,C,y),sol)

def influence(BETAS,ntop):
    means = np.mean(abs(BETAS),axis=0)
    top = np.argpartition(means, -ntop)[-ntop:]
    return(np.sort(top))

def plot_betai(labels,l_index,path,BETAS):
    j=0
    for i in range(len(BETAS[0])) :
        if(j<len(l_index) and i==l_index[j]):
            if not (type(labels)==bool): leg = 'Coefficient '+str(labels[i])
            else : leg = 'Coefficient '+str(i)
            plt.plot(path,BETAS[:,i],label=leg,color=colo[j])
            j+=1
        else:
            plt.plot(path, BETAS[:, i], color=colo[i+j])

def affichage(LISTE_BETA, path, title=' ', labels=False, pix=False, xlabel=" ", ylabel=" ", naffichage=10):
    BETAS = np.array(LISTE_BETA)
    l_index = influence(BETAS, naffichage)
    plt.figure(figsize=(10, 3), dpi=80)
    if (pix == 'path'): plt.plot(path, [0] * len(path), 'r+')
    plot_betai(labels, l_index, path, BETAS)
    plt.title(title), plt.legend(loc=4, borderaxespad=0.)
    plt.xlabel(xlabel), plt.ylabel(ylabel)
    if (type(pix) == bool and pix == True):
        plt.matshow([[(abs(LISTE_BETA[i][j]) > 1e-2) for i in range(len(LISTE_BETA))] for j in
                     range(len(LISTE_BETA[0]))]),plt.show()


def normalize(lb,lna,ly):
    for j in range(len(lb[0])):
        lb[:,j] =  lb[:,j]*ly/lna[j]
    return(lb)     

def denorm(B,lna,ly): return(np.array([ly*B[j]/(np.sqrt(len(B))*lna[j]) for j in range(len(B))]) ) 





def csv_to_mat(file,begin = 1, header=None):
    ''' Function to read a csv file and to create an ndarray with this

    Args:
        file (str): Name of csv file
        begin (int, optional): First colomn where it should read the matrix
        header (None or int, optional): Same parameter as in the function :func:`pandas.read_csv`

    Returns:
        ndarray : matrix of the csv file
    '''
    tab1=pd.read_csv(file,header=header)
    return(np.array(tab1)[:,begin:])

def rescale(matrices):
    ''' Function that rescale the matrix and returns its scale

    Substract the mean of y, then divides by its norm. Also divide each colomn of X by its norm.
    This will change the solution, not only by scaling it, because then the L1 norm will affect every component equally (and not only the variables with big size)

    Args:
        matrices (tuple) : tuple of three ndarray matrices corresponding to (X,C,y)

    Returns:
        tuple : tuple of the three corresponding matrices after normalization
        tuple : tuple of the three information one need to recover the initial data : lX (list of initial colomn-norms of X), ly (initial norm of y), my (initial mean of y)

    '''
    (X,C,y)=matrices
    my = sum(y)/len(y)
    lX = [LA.norm(X[:,j]) for j in range(len(X[0]))]
    ly = LA.norm(y-my*np.ones(len(y)))
    Xn = np.array([[X[i,j]/(lX[j])  for j in range(len(X[0]))] for i in range(len(X))])
    yn = np.array([ (y[j]-my)/ly for j in range(len(y)) ])
    Cn = np.array([[C[i,j]*ly/lX[j] for j in range(len(X[0]))] for i in range(len(C))])
    return((Xn,Cn,yn),(lX,ly,my))


def hub(r,rho) : 
    h=0
    for j in range(len(r)):
        if(abs(r[j])<rho): h+=r[j]**2
        elif(r[j]>0)     : h+= (2*r[j]-rho)*rho
        else             : h+= (-2*r[j]-rho)*rho
    return(h)
def L_LS(A,y,lamb,x): return(LA.norm( A.dot(x) - y )**2 + lamb * LA.norm(x,1))
def L_conc(A,y,lamb,x): return(LA.norm( A.dot(x) - y ) + np.sqrt(2)*lamb * LA.norm(y,1))
def L_H(A,y,lamb,x,rho): return(hub( A.dot(x) - y , rho) + lamb * LA.norm(x,1))


# Compute I - C^t (C.C^t)^-1 . C : the projection on Ker(C)
def proj_c(M,d):
    if (LA.matrix_rank(M)==0):  return(np.eye(d))
    return(np.eye(d)-LA.multi_dot([M.T,np.linalg.inv(M.dot(M.T) ),M]) )


def theoretical_lam(n,d):
    ''' Theoretical lambda as a function of the dimension of the problem

    This function returns (with :math:`\phi = erf`) :

    :math:`4/ \sqrt{n}  \phi^{-1}(1 - 2x)` such that  :math:`x = 4/d ( \phi^{-1}(1-2x)4 + \phi^{-1}(1-2x)^2 )`

    Which is the same (thanks to formula : :math:`norm^{-1}(1-t) = \sqrt{2}\phi^{-1}(1-2t)` ) as :

    :math:`\sqrt{2/n} * norm^{-1}(1-k/p)` such that  :math:`k = norm^{-1}(1 - k/p)^4 + 2norm^{-1}(1 - k/p)^2`

    Args:
        n (int) : number of sample
        d (int)  : number of variables

    Returns:
        float : theoretical lambda

    '''
    x=0.
    dx = 0.1
    for i in range(10):
        bo = True
        while bo :
            x    += dx
            f     = erfinv(1-2*x)
            xd    = 4/d * (f**4+f**2)
            bo    = (xd>x)           
        x = x-dx
        dx = dx/10
    return(2*f/np.sqrt(n))

# function to do LS : return  X (X^t X)^-1  X^t y
def min_LS(matrices,selected):
    X,C,y = matrices
    Xr, Cr = X[:,selected],C.T[selected]
    proj = np.eye(len(Cr)) - Cr.dot(LA.pinv(Cr))
    ls = LA.multi_dot([ proj, LA.pinv(Xr.dot(proj)),y]) 
    beta = np.zeros(len(X[0]))
    beta[selected] = ls
    return(beta)

def clr(array, coef=0.5):
    ''' Centered-Log-Ratio transformation

    Set all negaitve or null entry to a constant coef. Then compute the log of each component. Then substract the mean of each colomn on each colomn.

    Args:
        array (ndarray) : matrix nxd
        coef (float, optional)  : Value to replace the zero values

    Returns:
        ndarray : clr transformed matrix nxd

    '''
    M = np.copy(array)
    null_set = (M <= 0.)
    M[null_set] = np.ones(M[null_set].shape)*coef
    M = np.log(M)
    return(M - np.mean(M, axis=0))

def mat_to_np(file):
    ''' Function to read a mat file and to create an ndarray with this

    Args:
        file (str): Name of mat file

    Returns:
         ndarray : matrix of the mat file
    '''
    arrays = {}
    f = h5py.File(file)
    for k,v in f.items():
        arrays[k]=np.array(v)
    return arrays


'''
def verify(obj,beta,sigma):
    epsilon = 0.01
    N = 1000
    booleans = np.array([True]*N)
    opt = obj(beta,sigma)
    for i in range(N):
        beta_prime = beta[:]
        for j in range(len(beta_prime)):
            beta_prime[j]+=epsilon*np.random.random()
        sigma_prime = sigma + epsilon*np.random.random()
        booleans[i] = (obj(beta_prime,sigma_prime) > opt)
    return(np.all(booleans))


def plot_influence(labels,l_influence):
    print("influent variables  : \n \n")
    for beta in l_influence:
        string=''
        for colo in range(1,7):
            string+= str(labels.loc[beta,colo])
        print(string + '\n')

def support_dist(x,y):
    s = 0 
    n = len(x)
    for k in range(n):
        x_null, y_null = (abs(x[k])<1e-4), (abs(y[k])<1e-4)
        if (x_null and not y_null) or (y_null and not x_null): s+=1
    return(s)
        
def compare(L,show,lam):
    str1 = 'Execution time : \n \n'
    str2 = '\n Iterations : \n \n'
    str3 = '\n L2-difference between : \n \n'
    str4 = '\n For concomitant algorithms : \n \n'
    for i in range(len(L)) :
        x,niter,dt = L[i][0][:3]
        name = L[i][1]
        if (show=='sep' or show==True):
            plt.title('lam = '+str(lam))
            plt.bar(range(len(x)),x,label=name,color=colo[i])
            plt.legend()
            if (show=='sep'): plt.show()
        if (niter != 0):
            str1+= name+' :'+str(round(dt,5)) + '\n'
            str2+= name+' :'+str(niter)+ '\n'
        for j in range (i+1,len(L)): str3+= name + ' and '+L[j][1] +'             : ' + str(round(LA.norm((L[j][0][0]-x)),4)) + '\n'
    plt.show()
    printsigma =False
    for i in range(len(L)) :
        if (len(L[i][0])==4): 
            printsigma = True
            str4+=' Sigma '+L[i][1]+'  :'+str(round(L[i][0][3],3))
 
    print(str1)
    print(str2)
    if (len(L)>1): print(str3)
    if (printsigma): print(str4)
    
    
def old_normalize(lb,lna,ly):
    lbetaprime = []
    for beta in lb:
        lbetaprime.append( np.array([ly*beta[j]/(lna[j]) for j in range(len(beta))]) )
    return(lbetaprime)
    
def name(tol,string): return(string+'  tol={:.0e}'.format(float(tol)))   

def aff(M):
    (m,n)= M.shape
    for i in range(m):
        string =  ''
        for j in range(n):
            to_ad=str(round(abs(M[i,j]),3))
            string+=to_ad+' '*(5-len(to_ad))+' |'
        print(string + '//')   
        
def influence_old(sol,top):
    l_influence = []
    for j in range(top):
        maxi = 0.
        for i in range(len(sol)): 
            if ( not (i in l_influence) and (abs(sol[i])>=maxi)): maxi,argmaxi = abs(sol[i]), i
        l_influence.append(argmaxi)
    return(l_influence)
    
    
def plot_betai_old(labels,l_index,path,BETA):
    for i in range(len(l_index)) :
        leg = 'index = '+str(l_index[i])
        if not (type(labels)==bool): leg+='  '+ str(labels[l_index[i]])
        if (i>10): plt.plot(path,[BETA[ilam][l_index[i]] for ilam in range(len(path))],color=colo[l_index[i]])
        else:     plt.plot(path,[BETA[ilam][l_index[i]] for ilam in range(len(path))],label=leg,color=colo[l_index[i]])
'''