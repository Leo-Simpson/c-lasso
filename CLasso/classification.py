N=10000
import numpy as np
import numpy.linalg as LA
from CLasso.general_path_alg import pathalgo_general, solve_path

def solve_cl_path(matrices,lamin, n_active):
    return (solve_path(matrices,lamin,up, n_active,0))

def pathalgo_cl(matrix,path,n_active=False):
    return(pathalgo_general(matrix,path,up,n_active))



#function that search the next lambda where something happen, and update the solution Beta
def up(param):
    lambdamax,lamin,A, y  = param.lambdamax,param.lamin,param.A, param.y
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
        Xt = LA.inv(M[activity + idr, :][:, activity + idr])
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
                Xt = LA.inv(M[activity + idr, :][:, activity + idr])

        param.number_act, param.idr, param.Xt, param.activity, param.F, param.beta, param.s, param.lam, param.M, param.r = \
        number_act, idr, Xt, activity, F, beta, s, lam, M, r



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

#When we remove an active parameter
def next_idr2(liste,mat):
    if(sum(liste)==0): return(False)
    R = LA.qr(mat[liste,:].T)[1]
    for i in range(len(R)):
        if(abs(R[i,i])<1e-10):             # looking for the i-th True element of liste
            j,somme = 0, liste[0]
            while(somme<=i):
                j, somme = j+1, somme + liste[j+1]
            return(j)
    return(False)

#Update the invers of a matrix whom we add a line, which is useul to compute the derivatives
def next_inv(Xt,B,al,ligne):
    n=len(Xt)
    Yt = np.zeros((n+1,n+1))
    alpha = 1/al
    B = np.array([B])
    b1 = Xt[:ligne,:][:,:ligne]+ alpha*B[:,:ligne].T.dot(B[:,:ligne])
    b2 = Xt[ligne:,:][:,:ligne]+ alpha*B[:,ligne:].T.dot(B[:,:ligne])
    b4 = Xt[ligne:,:][:,ligne:]+ alpha*B[:,ligne:].T.dot(B[:,ligne:])
    col1 = np.concatenate((b1,-alpha*B[:,:ligne],b2), axis = 0)
    col2 = np.concatenate((b2.T,-alpha*B[:,ligne:],b4), axis = 0)
    col = np.concatenate((-alpha*B[0,:ligne],[alpha],-alpha*B[0,ligne:]), axis = 0)
    return(np.concatenate((col1,np.array([col]).T,col2), axis = 1))
