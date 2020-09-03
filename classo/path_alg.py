N = 10000
N_frac = 100
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
    def __init__(self,matrices,lamin, rho,typ, eps_L2 = 1e-3):

        (self.A, self.C, self.y) = matrices
        self.lamin = lamin
        self.rho = rho
        self.formulation = typ
        n, d, k = len(self.A), len(self.A[0]), len(self.C)
        self. number_act = 0
        self.idr         = [False] * k
        self.activity    = [False] * d
        self.beta        = np.zeros(d)
        s                = 2*self.A.T.dot(h_prime(self.y,rho))
        self.lambdamax   = LA.norm(s, np.inf)
        self.lamin       = lamin
        self.s           = s/self.lambdamax
        self.lam         = 1.
        self.r           = -self.y
        self.F           = [True] * n
        self.eps_L2      = eps_L2
        if (rho>0):
            for j in range(n):
                if( abs(self.y[j]) > self.rho ): self.F[j] = False
        elif (rho<0):
            for j in range(n):
                if( self.y[j] < self.rho ): self.F[j] = False
        AtA = self.A[self.F].T.dot(self.A[self.F]) + self.eps_L2*np.eye(d)
        for i in range(d):
            if (self.s[i] == 1. or self.s[i] == -1.):
                self.activity[i] = True
                self.number_act += 1
                if (k > 0):
                    to_ad = next_idr1(self.idr, self.C[:, self.activity])
                    if (type(to_ad) == int): self.idr[to_ad] = True


        if(k==0):  self.M = 2 * AtA
        else    :  self.M  = np.concatenate((np.concatenate((2 * AtA, self.C.T), axis=1), np.concatenate((self.C, np.zeros((k, k))), axis=1)),axis=0)


        try : 
            self.Xt = LA.inv(self.M[self.activity + self.idr, :][:, self.activity + self.idr])
        except LA.LinAlgError :
            self.Xt = LA.inv( self.M[self.activity + self.idr, :][:, self.activity + self.idr] + np.diag([self.eps_L2]*sum(self.activity) + [0]*sum(self.idr) )   )


# iteration of the function up to solve the path at each breaking points.
def solve_path(matrices, lamin, n_active, rho,typ):
    d = len(matrices[0][0])
    param = parameters_for_update(matrices, lamin, rho,typ)
    BETA, LAM = [param.beta], [param.lam]
    if param.lam < lamin : return BETA,LAM
    for i in range(d*N_frac):
        up(param)
        BETA.append(param.beta), LAM.append(param.lam)
        if (n_active > 0  and param.number_act >= n_active) or param.lam == lamin : 
            return (BETA, LAM)

    raise ValueError("The path algorithm did not finsh after %i iterations " %N, "and with lamin=%f" %lamin, " and with n_active ="+str(n_active))

def solve_path_Conc(matrices, stop, n_active=False, lassopath=True,true_lam=False):
    (A, C, y) = matrices
    n, d, k = len(A), len(A[0]), len(C)
    # to compute r = (A beta - y)/||y|| more efficientely ; and we set reduclam=lam/stop to 2 so that if stop = 0, the condition reduclam < ||r|| is never furfilled
    A_over_NORMy, y_over_NORMy, reduclam = A / (LA.norm(y)), y / (LA.norm(y)), 2.
    if (lassopath):
        lamin, R = 0, [-y_over_NORMy]
    else:
        lamin, beta_old, reduclam_old, r_old = 0, np.zeros(d), 1., -y_over_NORMy

    param = parameters_for_update(matrices, lamin, 0, "Conc")
    BETA, LAM = [param.beta], [param.lam]
    for i in range(d*N_frac):

        up(param)
        BETA.append(param.beta), LAM.append(param.lam)
        param.r = A_over_NORMy.dot(param.beta) - y_over_NORMy
        if (stop != 0): reduclam = param.lam / stop
        if (lassopath):
            R.append(param.r)
            if (reduclam <= LA.norm(param.r)) or (param.number_act >= n - k) or (
                    n_active > 0 and param.number_act >= n_active): return (BETA, LAM, R)
        else:
            if reduclam <= LA.norm(param.r): return ((beta_old, param.beta), (reduclam_old, reduclam), (r_old, param.r))
            beta_old, reduclam_old, r_old = param.beta, reduclam, param.r

    raise ValueError("The concomitant path algorithm did not finsh after %i iterations " %N, "and with lamin=" + str(stop), " and with n_active ="+str(n_active))





# Functions to interpolate the solution path between the breaking points
def pathalgo_general(matrix,path,typ,n_active=False,rho=0):
    BETA, i = [], 0
    X,sp_path = solve_path(matrix,path[-1],n_active,rho,typ)
    sp_path.append(path[-1]),X.append(X[-1])
    for lam in path:
        while (lam<sp_path[i+1]): i+=1
        teta = (sp_path[i]-lam)/(sp_path[i]-sp_path[i+1])
        BETA.append(X[i]*(1-teta)+X[i+1]*teta)
    return(BETA)

def pathalgo_huber_cl(matrix, path, rho, n_active=False):
    return( pathalgo_general(matrix, path, 'huber_cl', n_active=n_active, rho=rho) )

def pathalgo_cl(matrix,path,n_active=False):
    return(pathalgo_general(matrix,path,'cl',n_active))


def up(param):
    formulation = param.formulation
    if (formulation in ['LS','Conc']  ):
        up_LS(param)
    elif (formulation == 'huber'):
        up_huber(param)
    elif (formulation == 'cl'):
        up_cl(param)
    elif (formulation == 'huber_cl'):
        up_huber_cl(param)


# function that search the next lambda where something happen, and update the solution Beta
def up_LS(param):
    lambdamax, lamin, M, C, eps_L2 = param.lambdamax, param.lamin, param.M, param.C, param.eps_L2
    number_act, idr, Xt, activity, beta, s, lam = param.number_act, param.idr, param.Xt, param.activity, param.beta, param.s, param.lam

    d = len(activity)
    L = [lam] * d
    D, E = direction(activity, s, M[:len(activity), :][:, :len(activity)], M[d:, :][:, :d], Xt, idr, number_act)
    for i in range(d):
        bi, di, e, s0 = beta[i], D[i], E[i], s[i]
        if (activity[i]):
            if (abs(bi * di) > 1e-10 and bi * di < 0):
                L[i] = -bi / (di * lambdamax)
        else:
            if (abs(e - s0) < 1e-10): continue
            if (e > s0):
                dl = (1 + s0) / (1 + e)
            else:
                dl = (1 - s0) / (1 - e)
            L[i] = dl * lam
    dlamb = min(min(L), lam - lamin)
    # Update matrix inverse, list of rows in C and activity
    for i in range(d):
        if (L[i] < dlamb + 1e-10):
            if (activity[i]):
                activity[i], number_act = False, number_act - 1
                if (len(M) > d):
                    to_ad = next_idr2(idr, C[:, activity])
                    if (type(to_ad) == int): idr[to_ad] = False
            else:
                #x = M[:, activity + idr][i]
                #al = M[i, i] - np.vdot(x, Xt.dot(x))
                #if (abs(al) < 1e-10): break
                activity[i], number_act = True, number_act + 1
                if (len(M) > d):
                    to_ad = next_idr1(idr, C[:, activity])
                    if (type(to_ad) == int): idr[to_ad] = True
        
            
    try : 
        Xt = LA.inv(M[activity + idr, :][:, activity + idr])
    except LA.LinAlgError :
        Xt = LA.inv( M[activity + idr, :][:, activity + idr] + np.diag([eps_L2]*(sum(activity) + sum(idr) )) )
            

    beta = beta + lambdamax * D * dlamb
    if not (lam == dlamb): s = E + lam / (lam - dlamb) * (s - E)
    lam -= dlamb

    param.number_act, param.idr, param.Xt, param.activity, param.beta, param.s, param.lam = number_act, idr, Xt, activity, beta, s, lam

def up_huber(param):
    lambdamax, lamin, A, y, C, rho, eps_L2 = param.lambdamax, param.lamin, param.A, param.y, param.C, param.rho, param.eps_L2
    number_act, idr, Xt, activity, F, beta, s, lam, M, r = param.number_act, param.idr, param.Xt, param.activity, param.F, param.beta, param.s, param.lam, param.M, param.r
    d = len(activity)
    L = [lam] * d
    D, E = direction(activity, s, M[:len(activity), :][:, :len(activity)], M[d:, :][:, :d], Xt, idr, number_act)
    for i in range(d):
        bi, di, e, s0 = beta[i], D[i], E[i], s[i]
        if (activity[i]):
            if (abs(bi * di) > 1e-10 and bi * di < 0):
                L[i] = -bi / (di * lambdamax)
        else:
            if (abs(e - s0) < 1e-10 or abs(s0) > 1): continue
            if (e > s0):
                dl = (1 + s0) / (1 + e)
            else:
                dl = (1 - s0) / (1 - e)
            L[i] = dl * lam

    dlamb = min(min(L), lam, lam - lamin)
    huber_up, ADl = False, A.dot(D) * lambdamax
    j_switch = None
    for j in range(len(r)):
        #     find if there is a  0< dl < dlamb such that F[j] and |r[j]+ADl[j]*dl|>rho
        # or  find if there is a  0< dl < dlamb such that not F[j] |r[j]+ADl[j]*dl|<rho
        if (abs(ADl[j]) < 1e-6 or abs(r[j]) < 1e-6 or abs(abs(r[j]) - rho) < 1e-6): continue
        c = -r[j] / ADl[j]
        dl = (1 - rho / abs(r[j])) * c
        if (dl < 0):
            if (c > 0):
                dl = (1 + rho / abs(r[j])) * c
            else:
                dl = dlamb
        if (dl < dlamb):
            huber_up, j_switch, dlamb = True, j, dl
    beta, s, r, lam = beta + lambdamax * D * dlamb, E + lam / (lam - dlamb) * (s - E), r + ADl * dlamb, lam - dlamb

    if (huber_up):
        F[j_switch] = not F[j_switch]
        M[:d, :][:, :d] = 2 * A[F].T.dot(A[F]) + eps_L2*np.eye(d)
    else:
        # Update matrix inverse, list of rows in C and activity
        for i in range(d):
            if (L[i] < dlamb + 1e-10):
                if (activity[i]):
                    activity[i], number_act = False, number_act - 1
                    if (len(M) > d):
                        to_ad = next_idr2(idr, M[d:, :][:, :d][:, activity])
                        if (type(to_ad) == int): idr[to_ad] = False
                else:
                    activity[i], number_act = True, number_act + 1
                    if (len(M) > d):
                        to_ad = next_idr1(idr, M[d:, :][:, :d][:, activity])
                        if (type(to_ad) == int): idr[to_ad] = True

    try : 
        Xt = LA.inv(M[activity + idr, :][:, activity + idr])
    except LA.LinAlgError :
        Xt = LA.inv( M[activity + idr, :][:, activity + idr] + np.diag([eps_L2]*(sum(activity) + sum(idr) )) )


    param.number_act, param.idr, param.Xt, param.activity, param.F, param.beta, param.s, param.lam, param.M, param.r = number_act, idr, Xt, activity, F, beta, s, lam, M, r

def up_cl(param):
    lambdamax,lamin,A, y, eps_L2  = param.lambdamax,param.lamin,param.A, param.y, param.eps_L2
    number_act,idr,Xt,activity,F,beta,s, lam, M,r =\
        param.number_act,param.idr,param.Xt,param.activity,param.F,param.beta,param.s, param.lam, param.M,param.r

    d = len(activity)
    L = [lam] * d
    D, E = direction(activity, s, M[:len(activity), :][:, :len(activity)], M[d:, :][:, :d], Xt, idr, number_act)
    for i in range(d):
        bi, di, e, s0 = beta[i], D[i], E[i], s[i]
        if (activity[i]):
            if (abs(bi * di) > 1e-10 and bi * di < 0):
                L[i] = -bi / (di * lambdamax)
        else:
            if (abs(e - s0) < 1e-10 or abs(s0) > 1): continue
            if (e > s0):
                dl = (1 + s0) / (1 + e)
            else:
                dl = (1 - s0) / (1 - e)
            L[i] = dl * lam

    dlamb = min(min(L), lam, lam - lamin)
    max_up, yADl = False, y*A.dot(D) * lambdamax
    j_switch = None
    for j in range(len(r)):
        #     find if there is a  0< dl < dlamb such that F[j] and r[j]+yADl[j]*dl > 1
        # or  find if there is a  0< dl < dlamb such that not F[j] r[j]+yADl[j]*dl < 1
        if (abs(r[j]-1)<1e-4): continue
        if (yADl[j] != 0.): dl = (1-r[j]) / yADl[j]
        else : dl = -1
        if (dl < dlamb and dl >0):
            max_up, j_switch, dlamb = True, j, dl

    beta, s, r, lam = beta + lambdamax * D * dlamb, E + lam / (lam - dlamb) * (s - E), r + yADl * dlamb, lam - dlamb

    if (max_up):
        F[j_switch] = not F[j_switch]
        M[:d, :][:, :d] = 2 * A[F].T.dot(A[F])
    else:
        # Update matrix inverse, list of rows in C and activity
        for i in range(d):
            if (L[i] < dlamb + 1e-10):
                if (activity[i]):
                    activity[i], number_act = False, number_act - 1
                    if (len(M) > d):
                        to_ad = next_idr2(idr, M[d:, :][:, :d][:, activity])
                        if (type(to_ad) == int): idr[to_ad] = False
                else:
                    # x = M[:,activity+idr][i]
                    # al = M[i,i]-np.vdot(x,Xt.dot(x))
                    # if (abs(al)<1e-10): break
                    activity[i], number_act = True, number_act + 1
                    if (len(M) > d):
                        to_ad = next_idr1(idr, M[d:, :][:, :d][:, activity])
                        if (type(to_ad) == int): idr[to_ad] = True
    try : 
        Xt = LA.inv(M[activity + idr, :][:, activity + idr])
    except LA.LinAlgError :
        Xt = LA.inv( M[activity + idr, :][:, activity + idr] + np.diag([eps_L2]*(sum(activity) + sum(idr) )) )

    param.number_act, param.idr, param.Xt, param.activity, param.F, param.beta, param.s, param.lam, param.M, param.r = \
        number_act, idr, Xt, activity, F, beta, s, lam, M, r

def up_huber_cl(param):
    lambdamax, lamin, A, y, rho, eps_L2 = param.lambdamax, param.lamin, param.A, param.y, param.rho, param.eps_L2
    number_act, idr, Xt, activity, F, beta, s, lam, M, r = param.number_act, param.idr, param.Xt, param.activity, param.F, param.beta, param.s, param.lam, param.M, param.r
    d = len(activity)
    L = [lam] * d
    D, E = direction(activity, s, M[:len(activity), :][:, :len(activity)], M[d:, :][:, :d], Xt, idr, number_act)
    for i in range(d):
        bi, di, e, s0 = beta[i], D[i], E[i], s[i]
        if (activity[i]):
            if (abs(bi * di) > 1e-10 and bi * di < 0):
                L[i] = -bi / (di * lambdamax)
        else:
            if (abs(e - s0) < 1e-10 or abs(s0) > 1): continue
            if (e > s0):
                dl = (1 + s0) / (1 + e)
            else:
                dl = (1 - s0) / (1 - e)
            L[i] = dl * lam

    dlamb = min(min(L), lam, lam - lamin)
    max_up, yADl = False, y * A.dot(D) * lambdamax
    j_switch = None
    for j in range(len(r)):
        #     find if there is a  0< dl < dlamb such that F[j] and r[j]+yADl[j]*dl > 1
        # or  find if there is a  0< dl < dlamb such that not F[j] r[j]+yADl[j]*dl < 1
        #     find if there is a  0< dlhuber < dlamb such that F[j] and r[j]+yADl[j]*dl < rho
        # or  find if there is a  0< dlhuber < dlamb such that not F[j] r[j]+yADl[j]*dl > rho


        if (yADl[j] != 0.):
            dlmax      = (1 - r[j]) / yADl[j]
            dlhuber = (rho - r[j]) / yADl[j]
        else: dlmax,dlhuber = dlamb+1,dlamb+1
        if(dlmax<=0.):     dlmax   = dlamb+1
        if(dlhuber <= 0.): dlhuber = dlamb+1

        dl = min ( dlhuber , dlmax)

        if (dl < dlamb) : max_up, j_switch, dlamb = True, j, dl

    beta, s, r, lam = beta + lambdamax * D * dlamb, E + lam / (lam - dlamb) * (s - E), r + yADl * dlamb, lam - dlamb

    if (max_up):
        F[j_switch] = not F[j_switch]
        M[:d, :][:, :d] = 2 * A[F].T.dot(A[F])

    else:
        # Update matrix inverse, list of rows in C and activity
        for i in range(d):
            if (L[i] < dlamb + 1e-10):
                if (activity[i]):
                    activity[i], number_act = False, number_act - 1
                    if (len(M) > d):
                        to_ad = next_idr2(idr, M[d:, :][:, :d][:, activity])
                        if (type(to_ad) == int): idr[to_ad] = False
                else:
                    # x = M[:,activity+idr][i]
                    # al = M[i,i]-np.vdot(x,Xt.dot(x))
                    # if (abs(al)<1e-10): break
                    activity[i], number_act = True, number_act + 1
                    if (len(M) > d):
                        to_ad = next_idr1(idr, M[d:, :][:, :d][:, activity])
                        if (type(to_ad) == int): idr[to_ad] = True

    try : 
        Xt = LA.inv(M[activity + idr, :][:, activity + idr])
    except LA.LinAlgError :
        Xt = LA.inv( M[activity + idr, :][:, activity + idr] + np.diag([eps_L2]*(sum(activity) + sum(idr) )) )

    param.number_act, param.idr, param.Xt, param.activity, param.F, param.beta, param.s, param.lam, param.M, param.r = number_act, idr, Xt, activity, F, beta, s, lam, M, r


















# Compute the derivatives of the solution Beta and the derivative of lambda*subgradient thanks to the ODE
def direction(activity,s,Mat,C,Xt,idr,number_act):
    if (len(C)==0):
        D,product =np.zeros(len(activity)), Xt[:,:number_act].dot(s[activity])
        D[activity]= product
        return(D,Mat.dot(D))
    D,Dadj=np.zeros(len(activity)),np.zeros(len(C))
    product = Xt[:,:number_act].dot(s[activity])
    D[activity],Dadj[idr]=product[:number_act],product[number_act:]
    E = (Mat.dot(D)+C.T.dot(Dadj))   #D and D2 in Rd with zeros in inactives and E. D is - derivatives
    return(D,E)

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

# When we remove an active parameter
def next_idr2(liste, mat):
    if (sum(liste) == 0): return (False)
    R = LA.qr(mat[liste, :].T)[1]
    for i in range(len(R)):
        if (abs(R[i, i]) < 1e-10):  # looking for the i-th True element of liste
            j, somme = 0, liste[0]
            while (somme <= i):
                j, somme = j + 1, somme + liste[j + 1]
            return (j)
    return (False)

# Update the invers of a matrix whom we add a line, which is useul to compute the derivatives
def next_inv(Xt, B, al, ligne):
    n = len(Xt)
    Yt = np.zeros((n + 1, n + 1))
    alpha = 1 / al
    B = np.array([B])
    b1 = Xt[:ligne, :][:, :ligne] + alpha * B[:, :ligne].T.dot(B[:, :ligne])
    b2 = Xt[ligne:, :][:, :ligne] + alpha * B[:, ligne:].T.dot(B[:, :ligne])
    b4 = Xt[ligne:, :][:, ligne:] + alpha * B[:, ligne:].T.dot(B[:, ligne:])
    col1 = np.concatenate((b1, -alpha * B[:, :ligne], b2), axis=0)
    col2 = np.concatenate((b2.T, -alpha * B[:, ligne:], b4), axis=0)
    col = np.concatenate((-alpha * B[0, :ligne], [alpha], -alpha * B[0, ligne:]), axis=0)
    return (np.concatenate((col1, np.array([col]).T, col2), axis=1))

def h_lambdamax(X, y, rho):
    return 2 * LA.norm(X.T.dot(h_prime(y, rho)), np.infty)

# Compute the derivative of the huber function, particulary useful for the computing of lambdamax
def h_prime(y,rho):
    if (rho==0): return(y)
    m = len(y)
    lrho = rho*np.ones(m)
    return(np.maximum(lrho,-y)+ np.minimum(y-lrho,0))




'''
Function that solve the concomitant problem for every lambda thanks to the previous function : we firstly compute all the non concomitant Least square problems, then we use it to find sigma and so the solution, using the equation for sigma: 

sigma = || A*B(lambda*sigma) - y ||_2       (where B(lambda) is found thanks to solve_path)

            teta = (sp_path[i]-lam)/(sp_path[i]-sp_path[i+1])
'''
