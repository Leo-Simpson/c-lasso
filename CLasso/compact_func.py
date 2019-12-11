from time import time
import numpy as np
import matplotlib.pyplot as plt

from CLasso.solve_LS import problem_LS, algo_LS, pathalgo_LS
from CLasso.solve_Huber import problem_Huber, algo_Huber, pathalgo_Huber
from CLasso.solve_Concomitant import problem_Concomitant, algo_Concomitant, pathalgo_Concomitant
from CLasso.solve_Concomitant_Huber import problem_Concomitant_Huber, algo_Concomitant_Huber, pathalgo_Concomitant_Huber
from CLasso.little_functions import affichage, random_data


'''
Classo and pathlasso are the main functions, they can call every algorithm acording to the method required
'''



def Classo(matrix,lam,typ = 'LS', meth='2prox',plot_time=True , plot_sol=True,plot_sigm=True , rho = 1.345, get_lambdamax = False):
    
        

        
        
    if(typ=='Concomitant'):
        
        t0 = time() 
        
        if not meth in ['ODE','2prox']:
            meth='2prox'
            if (lam>0.1): meth = 'ODE'        # use path algorithm if lam is high, but prox algo if lam is little

        pb = problem_Concomitant(matrix,meth)
        beta,s = algo_Concomitant(pb,lam)
        
        dt = time()-t0
    
    
    elif(typ=='Concomitant_Huber'):
        t0 = time() 
        
        if not meth in ['2prox']: meth='2prox'

        pb  = problem_Concomitant_Huber(matrix,meth,rho)
        beta,s = algo_Concomitant_Huber(pb,lam)
        
        dt = time()-t0
        
    elif(typ=='Huber'):
        
        t0 = time() 
        
        if not meth in ['ODE','2prox','FB','Noproj','cvx']:
            meth='FB'
            if (lam>0.1): meth = 'ODE'        # use path algorithm if lam is high, but prox algo if lam is little

        pb = problem_Huber(matrix,meth,rho)
        beta = algo_Huber(pb,lam)
        
        dt = time()-t0     
            
    else: 
       
        t0 = time()  
        
        if not meth in ['ODE','2prox','FB','Noproj','cvx']:
            meth='2prox'
            if (lam>0.1): meth = 'ODE'        # use path algorithm if lam is high, but prox algo if lam is little

        pb = problem_LS(matrix,meth)
        beta = algo_LS(pb,lam)
 
        dt = time()-t0
        
    if (plot_sigm and typ in ['Concomitant','Concomitant_Huber']): print('sigma = ',s)
    if (plot_time): print('Running time :', round(dt,5))
    if (plot_sol): plt.bar(range(len(beta)),beta),plt.title('Problem '+typ+' , lam ='+str(round(lam,3))+' solved with '+ meth +' method'),plt.savefig('Problem '+typ+' solved with '+ meth +' method'+'.png'), plt.show()
    if (typ  in ['Concomitant','Concomitant_Huber']): 
        if (get_lambdamax): return(pb.lambdamax,beta,s)
        else              : return(beta,s)
    if (get_lambdamax): return(pb.lambdamax,beta)
    else              : return(beta)




def pathlasso(matrix,lambdas=False,n_active=False,lamin=1e-2,typ='LS',meth='ODE',plot_time=True,plot_sol=True,plot_sigm=True,rho = 1.345, compare=False, true_lam = False ):

    if (type(compare)!= bool):
        diff,path = compare
        lambdas   = [lam/path[0] for lam in path]
    
    if (type(lambdas)!= bool):
        if (lambdas[0]<lambdas[-1]): lambdas = [lambdas[i] for i in range(len(lambdas)-1,-1,-1)]  # reverse the list needed
        lamin = lambdas[-1]
    else: lambdas = np.linspace(1.,lamin,100)

    if(typ=='Huber'):
        t0 = time()
        pb = problem_Huber(matrix,meth,rho)
        lambdamax = pb.lambdamax
        if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        sol  = pathalgo_Huber(pb,lambdas,n_active=n_active)
        if (type(sol)== tuple): BETA,LAM = sol
        else : BETA,LAM= sol,lambdas
        real_path = [lam*lambdamax for lam in LAM]
        dt = time()-t0
        
        
    elif(typ=='Concomitant'):
        
        t0 = time()
        pb = problem_Concomitant(matrix,meth)
        lambdamax = pb.lambdamax
        if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        sol  = pathalgo_Concomitant(pb,lambdas,n_active=n_active)
        if (len(sol)== 3): BETA,S,LAM = sol
        else :
            BETA,S= sol
            LAM = lambdas
        real_path = [lam*lambdamax for lam in LAM]
        dt = time()-t0
    
    
    elif(typ=='Concomitant_Huber'):
        t0 = time()
        meth='2prox'
        pb = problem_Concomitant_Huber(matrix,meth,rho)
        lambdamax = pb.lambdamax
        if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        BETA,S = pathalgo_Concomitant_Huber(pb,lambdas,n_active=n_active)
        real_path = [lam*lambdamax for lam in lambdas]
        dt = time()-t0
        
            
    else: 

        t0 = time()
        pb = problem_LS(matrix,meth)
        lambdamax = pb.lambdamax
        if (true_lam): lambdas=[lamb/lambdamax for lamb in lambdas]
        sol = pathalgo_LS(pb,lambdas,n_active=n_active)  #if n_active is not False, then the function will return a tuple with the list of lambdas
        if (type(sol) == tuple): BETA,LAM = sol
        else : BETA,LAM= sol,lambdas
        real_path = [lam*lambdamax for lam in LAM]
        dt = time()-t0





    if (plot_time): print('Running time :', round(dt,5))
    if (plot_sol): 
        affichage(BETA,real_path,title=typ+' Path for the method '+meth),plt.show()
        if not (type(compare)==bool): affichage([BETA[i]-diff[i] for i in range(len(BETA))],real_path,title='Difference between both methods'),plt.savefig('Difference between both methods'+'.png'),plt.show()
    if (plot_sigm and typ in ['Concomitant','Concomitant_Huber']):
        plt.plot(real_path,S),plt.ylabel("sigma / sigmaMAX"),plt.xlabel("lambda"), plt.title('Sigma for Concomitant'),plt.savefig('Sigma for Concomitant'+'.png'),plt.show()
        return(BETA,real_path,S)
    return(BETA,real_path)
 
    
    
    
    


# Cost fucntions for the three 'easiest' problems. Useful for test, to compare two solutions slightly different
def L_LS(A,y,lamb,x): return(LA.norm( A.dot(x) - y )**2 + lamb * LA.norm(x,1))
def L_conc(A,y,lamb,x): return(LA.norm( A.dot(x) - y ) + np.sqrt(2)*lamb * LA.norm(y,1))
def L_H(A,y,lamb,x,rho): return(hub( A.dot(x) - y , rho) + lamb * LA.norm(x,1))

def hub(r,rho) : 
    h=0
    for j in range(len(r)):
        if(abs(r[j])<rho): h+=r[j]**2
        elif(r[j]>0)     : h+= (2*r[j]-rho)*rho
        else             : h+= (-2*r[j]-rho)*rho
    return(h)




def ordre(meth):
    m,r_nonzero,k,sigma = 100,0.1,4,0.5
    D,T = range(50,500,20),[]
    for d in D:    
        s=0
        N_moy = d//10
        for i in range(N_moy):
            matrices,sol = random_data(m,d,int(d*r_nonzero),k,sigma)
            t0=time()
            X = Classo(matrices,0.5,meth=meth,plot_sol=False,plot_time=False)
            s+=time()-t0
            constr = LA.norm(matrices[1].dot(X))
            if (constr > 1e-10): print('C')
        T.append(s/N_moy)
    return(D,T)
