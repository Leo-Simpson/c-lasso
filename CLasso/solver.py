from time import time
import numpy as np
import matplotlib.pyplot as plt

from CLasso.little_functions import random_data, csv_to_mat, rescale, theoritical_lam
import scipy.io as sio
from CLasso.compact_func import Classo, pathlasso
from CLasso.cross_validation import CV
from CLasso.stability_selection import stability



class LAMparameters : 
    def __init__(self,n,d):
        self.lam              = 'theoritical'
        self.theoritical_lam  = round(theoritical_lam(n,d),3)
        self.formulation      = 'Concomitant'         
        # can also be : 'LS' ; 'Concomitant' ; 'Concomitant_Huber' or 'Huber'
        self.numerical_method = 'choose'            
        # can be : '2prox' ; 'ODE' ; 'Noproj' ; 'FB' ; and any other will make the algorithm decide
        


class CVparameters : 
    def __init__(self):
        
        self.seed             = 1
        self.formulation      = 'Concomitant'         
        # can also be : 'LS' ; 'Concomitant' ; 'Concomitant_Huber' or 'Huber'
        self.numerical_method = 'choose'            
        # can be : '2prox' ; 'ODE' ; 'Noproj' ; 'FB' ; and any other will make the algorithm decide
        
        self.Nsubset          = 5                       # Number of subsets used
        self.lamin            =  1e-2
        
        
        
class SSparameters :

    def __init__(self,n,d):
        
        self.seed = 1
        
        
        self.formulation      = 'Concomitant'         
        # can also be : 'LS' ; 'Concomitant' ; 'Concomitant_Huber' or 'Huber'
        self.numerical_method = 'choose'            
        # can be : '2prox' ; 'ODE' ; 'Noproj' ; 'FB' ; and any other will make the algorithm decide

        self.method           = 'first'      # Can be 'first' ; 'max' or 'lam'
        self.B                = 50
        self.q                = 10
        self.pourcent_nS      = 0.5
        self.lamin            = 1e-2          # the lambda where one stop for 'max' method
        self.hd               = False            # if set to True, then the 'max' will stop when it reaches n-k actives parameters
        self.lam              = 'theoritical'  # can also be a float, for the 'lam' method
        self.theoritical_lam  = round(theoritical_lam(int(n*self.pourcent_nS),d),4)
        

class classo_solution : 
    
    def __init__(self) : 
        self.beta_CV         = 'not computed'
        self.distribution_SS = 'not computed'
        self.beta_LAMfixed   = 'not computed'
        self.sigma_CV        = 'not computed'
        self.sigma_LAMfixed  = 'not computed'
        self.time_CV         = 0
        self.time_SS         = 0
        self.time_LAMfixed   = 0
        
    def __repr__(self):
        
        Xss, Xcv, Xlam = self.distribution_SS, self.beta_CV, self.beta_LAMfixed

        if (type(Xss) != str) :
            plt.bar(range(len(Xss)),Xss),   plt.title("Distribution of Stability Selection"), plt.show()
        
        if (type(Xcv) != str) :
            plt.bar(range(len(Xcv)),Xcv),   plt.title("Cross Validation"),                    plt.show()
        
        if (type(Xlam) != str) : 
            plt.bar(range(len(Xlam)),Xlam), plt.title("Solution for a fixed lambda"),         plt.show()
        
        return (    " Running time for CV       : "  + str(round(self.time_CV,3))       +
                "s \n Running time for SS       : "  + str(round(self.time_SS,3))       +
                "s \n Running time for LAMfixed : "  + str(round(self.time_LAMfixed,3)) +"s")
        
        

class classo_problem :
    
    def __init__(self,X,y):
        
        
        # Basic parameters
        
        n,d = len(X),len(X[0])
        self.C = np.ones((1,d))               #zero sum constraint by default, but it can be any matrix

        
        self.rescale = False               # booleen to know if we rescale the matrices
        
        self.rho = 1.345                  # only used for huber problems
        self.X = X
        self.y = y
        
        
        # Model selection parameters

        self.CV = True                             # Cross validation
        self.CVparameters = CVparameters()
        
        
        self.SS = True                             # Stability Selection
        self.SSparameters = SSparameters(n,d)

        
        self.LAMfixed = True              # Compute the solution for a fixed lambda
        self.LAMparameters = LAMparameters(n,d)
        
        
        # Plot parameters : 
        





    def solve(self):
        
        Lformulation = ['LS','Concomitant','Concomitant_Huber','Huber']
        
        
        X, C, y = self.X, self.C , self.y
        
        n,d      = len(y), len(X[0])
        matrices = (X,C,y)
        rho      = self.rho
        
        
        solution = classo_solution()
        
        if self.rescale : 
            matrices, self.scaling = rescale(matrices)         #SCALING contains  :
                                                        #(list of initial norms of A-colomns, 
                                                        #         initial norm of centered y,
                                                        #          mean of initial y )
        
        if self.CV : 
            
            t0 = time()
            param = self.CVparameters
            
            formulation           = param.formulation
            if not formulation in Lformulation:
                print("One of the formulations used does not exist yet on classo, please try one of the following : \n  -LS \n -Concomitant \n -Concomitant_Huber \n -Huber")
                return()
            
            
            numerical_method      = param.numerical_method
            K                     = param.Nsubset
            lamin                 = param.lamin
            seed                  = param.seed
            
            numerical_method = choose_numerical_method(numerical_method, 'CV', formulation)
            param.numerical_method = numerical_method
            
            
            output = CV(matrices,K,typ=formulation,meth=numerical_method,lamin=lamin,seed=seed)
            
            if len(output) == 2 :    solution.beta_CV, solution.sigma_CV = output   
            else :                   solution.beta_CV = output
              
            solution.time_CV = time()-t0


        if self.SS :
            
            t0 = time()
            param = self.SSparameters
            
            formulation = param.formulation
            if not formulation in Lformulation:
                print("One of the formulations used does not exist yet on classo, please try one of the following : \n  -LS \n -Concomitant \n -Concomitant_Huber \n -Huber")
                return()
            
            
            
            
            seed        = param.seed
            SSmethod    = param.method
            hd          = param.hd
            q           = param.q
            B           = param.B
            pourcent_nS = param.pourcent_nS
            

            if param.lam == 'theoritical' : lam = param.theoritical_lam
            else                          : lam = param.lam 
                
                
            numerical_method = choose_numerical_method(numerical_method,'SS', formulation,
                                                       SSmethod = SSmethod, lam = lam)
            param.numerical_method = numerical_method

            solution.distribution_SS = stability(matrices, SSmethod = SSmethod, numerical_method = numerical_method,
                                    lam=lam, hd = hd, q = q ,B = B, pourcent_nS = pourcent_nS,
                                    formulation = formulation,plot_time=False,seed=seed)
            

            solution.time_SS = time()-t0
            
            
        if self.LAMfixed : 
        
            t0 = time()
            param = self.LAMparameters
            
            formulation       = param.formulation
            if not formulation in Lformulation:
                print("One of the formulations used does not exist yet on classo, please try one of the following : \n  -LS \n -Concomitant \n -Concomitant_Huber \n -Huber")
                return()
            
            
            
            numerical_method  = param.formulation
            
            
            if param.lam == 'theoritical' : lam = param.theoritical_lam
            else                          : lam = param.lam 
            
            numerical_method = choose_numerical_method(numerical_method,'LAM', formulation, lam = lam)
            param.numerical_method = numerical_method
            
            
            output = Classo(matrices,lam,
                            typ = formulation, meth=numerical_method, 
                            plot_time=False , plot_sol=False, plot_sigm=False , rho = rho)
            
            if len(output) == 2 : solution.beta_LAMfixed, solution.sigma_LAMfixed = output  
            else                : solution.beta_LAMfixed = output
                
            solution.time_LAMfixed = time()-t0

        self.solution = solution
        
        

 




    
def choose_numerical_method(method,model,formulation,SSmethod = None, lam = None):
    
    if (formulation == 'Concomitant_Huber'):
        if not method in ['2prox']: return '2prox'
    
    
        
    # cases where we use classo at a fixed lambda    
    elif (model == 'LAM') or (model == 'SS' and SSmethod == 'lam') : 
        
        if formulation in ['Huber','LS']:
            if not method in ['ODE','2prox','FB','Noproj']:
                if (lam>0.1): return 'ODE'
                else        : return '2prox'            
        
        
        else :
            if not method in ['ODE','2prox']:
                if (lam>0.1): return 'ODE'
                else        : return '2prox'
      
    
    # cases where we use pathlasso                
    else:
        if formulation in ['Huber','LS']:
            if not method in ['ODE','2prox','FB','Noproj']: return 'ODE'         

        else :
            if not method in ['ODE','2prox']: return 'ODE'
            
    return method
                    

          
            
                    
                    
                    
                    
                    
                
                

