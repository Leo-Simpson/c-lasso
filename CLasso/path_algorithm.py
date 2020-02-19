N=100000
import numpy as np
import numpy.linalg as LA
from CLasso.general_path_alg import parameters_for_update

'''
Main function that solve the Least square problem for all lambda using the Path-Algo, which is describe in the paper : "Algorithms for Fitting the Constrained Lasso"

In all the code, capital letter variable are lists that contains the small variable 

'''




def solve_path(matrices,stop,n_active=False,return_lmax=False, concomitant = 'no'):

    (A, C, y) = matrices
    n, d, k = len(A), len(A[0]), len(C)

    if  (concomitant=='no')  : lamin  = stop
    else:
        # to compute r = (A beta - y)/||y|| more efficientely ; and we set reduclam=lam/stop to 2 so that if stop = 0, the condition reduclam < ||r|| is never furfilled
        A_over_NORMy, y_over_NORMy, reduclam = A/ (LA.norm(y))  , y / (LA.norm(y)), 2.
        if(concomitant=='path'): lamin,R = 0,[-y_over_NORMy]
        else                   : lamin,beta_old,reduclam_old,r_old = 0,np.zeros(d),1.,-y_over_NORMy

    param = parameters_for_update(matrices, lamin, 0)
    BETA, LAM = [param.beta], [param.lam]

    for i in range(N) :
        
        up(param)
        BETA.append(param.beta), LAM.append(param.lam)
        
        if not (concomitant=='no'):
            param.r = A_over_NORMy.dot(param.beta)-y_over_NORMy
            if (stop != 0) : reduclam = param.lam/stop
            if(concomitant=='path'): 
                R.append(param.r)
                if (reduclam <= LA.norm(param.r)) or (param.number_act >= n-k) or (type(n_active)==int and param.number_act>= n_active) : return(BETA,LAM,R)
            else : 
                if reduclam <= LA.norm(r): return((beta_old,beta),(reduclam_old,reduclam),(r_old,r))               
                beta_old,reduclam_old,r_old = beta,reduclam,r
            
        elif ((type(n_active)==int and number_act>= param.n_active) or param.lam == lamin):
            if(return_lmax):    return(BETA,LAM,param.lambdamax)
            else:               return(BETA,LAM)
            
    print('no conv')
    return(BETA,LAM)



#function that search the next lambda where something happen, and update the solution Beta
def up(param):
    lambdamax, lamin, M, C = param.lambdamax, param.lamin, param.M, param.C
    number_act, idr, Xt, activity, beta, s, lam = param.number_act, param.idr, param.Xt, param.activity, param.beta, param.s, param.lam
    
    d=len(activity)
    L = [lam]*d
    D,E = direction(activity,s,M[:len(activity),:][:,:len(activity)],M[d:,:][:,:d],Xt,idr,number_act)  
    for i in range(d):
        bi,di,e,s0 = beta[i],D[i],E[i],s[i]
        if (activity[i]): 
            if (abs(bi*di)>1e-10 and bi*di<0): 
                L[i]=-bi/(di*lambdamax)
        else :  
            if(abs(e-s0)<1e-10): continue
            if (e>s0): dl = (1+s0)/ (1+e)
            else: dl = (1-s0)/ (1-e)
            L[i] = dl * lam
    dlamb = min(min(L),lam-lamin)
    # Update matrix inverse, list of rows in C and activity
    for i in range(d):
        if (L[i]<dlamb+1e-10):
            if (activity[i]):
                activity[i], number_act = False, number_act - 1
                if(len(M)>d):
                    to_ad = next_idr2(idr,C[:,activity])
                    if(type(to_ad)==int): idr[to_ad] = False
            else:
                x = M[:,activity+idr][i]
                al = M[i,i]-np.vdot(x,Xt.dot(x))
                if (abs(al)<1e-10): break
                activity[i],number_act = True, number_act+1
                if(len(M)>d): 
                    to_ad = next_idr1(idr,C[:,activity])
                    if(type(to_ad)==int): idr[to_ad] = True
            
            
            Xt  = LA.inv(M[activity+idr,:] [:,activity+idr]) 
            
    beta = beta + lambdamax*D * dlamb
    if not (lam==dlamb): s = E + lam/(lam-dlamb) * (s-E)
    lam -= dlamb

    param.number_act, param.idr, param.Xt, param.activity, param.beta, param.s, param.lam = number_act, idr, Xt, activity, beta, s, lam


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


  

'''
Function that solve the concomitant problem for every lambda thanks to the previous function : we firstly compute all the non concomitant Least square problems, then we use it to find sigma and so the solution, using the equation for sigma: 

sigma = || A*B(lambda*sigma) - y ||_2       (where B(lambda) is found thanks to solve_path)

            teta = (sp_path[i]-lam)/(sp_path[i]-sp_path[i+1])
'''
    

    
    
    
    
