from time import time
import numpy as np
import matplotlib.pyplot as plt

from CLasso.little_functions import random_data, csv_to_mat, rescale, theoritical_lam
import scipy.io as sio
from CLasso.compact_func import Classo, pathlasso
from CLasso.cross_validation import CV
from CLasso.stability_selection import stability





        
        

        
        
        
class classo_problem :
    
    def __init__(self,X,y,C='zero-sum'): #zero sum constraint by default, but it can be any matrix

        n,d = len(X),len(X[0])

        
        ''' define the class classo_data that contains the data '''
        class classo_data : 
            def __init__(self,X,y,C):
                self.rescale = False               # booleen to know if we rescale the matrices
                self.X = X
                self.y = y
                if C=='zero-sum' : C = np.ones( (1,len(X[0])) )
                self.C = C
        ''' End of the definition''' 
        
        self.data = classo_data(X,y,C)
        self.rho = 1.345                  # only used for huber problems
        self.formulation      = 'Concomitant'
        '''
        define the class model_selection inside the class classo_problem
        '''
        class model_selection :
            def __init__(self,n,d):
                
                # Model selection parameters

                ''' CROSS VALIDATION PARAMETERS'''
                self.CV = True                            
                class CVparameters : 
                    def __init__(self):

                        self.seed             = 1
                        self.formulation      = 'not specified'     
                        # can also be : 'LS' ; 'Concomitant' ; 'Concomitant_Huber' or 'Huber'
                        self.numerical_method = 'choose'            
                        # can be : '2prox' ; 'ODE' ; 'Noproj' ; 'FB' ; and any other will make the algorithm decide

                        self.Nsubset          = 5                       # Number of subsets used
                        self.lamin            =  1e-2
                ''' End of the definition''' 
                
                self.CVparameters = CVparameters()

                
                
                ''' STABILITY SELECTION PARAMETERS'''
                self.SS = True 
                class SSparameters :
                    def __init__(self,n,d):
                        self.seed = 1
                        self.formulation      = 'not specified'      
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
                ''' End of the definition''' 
                
                self.SSparameters = SSparameters(n,d)


                
                
                
                ''' PROBLEM AT A FIXED LAMBDA PARAMETERS'''
                self.LAMfixed = True              
                class LAMparameters : 
                    def __init__(self,n,d):
                        self.lam              = 'theoritical'
                        self.theoritical_lam  = round(theoritical_lam(n,d),3)
                        self.formulation      = 'not specified'      
                        # can also be : 'LS' ; 'Concomitant' ; 'Concomitant_Huber' or 'Huber'
                        self.numerical_method = 'choose'            
                        # can be : '2prox' ; 'ODE' ; 'Noproj' ; 'FB' ; and any other will make the algorithm decide
                ''' End of the definition''' 
                
                self.LAMparameters = LAMparameters(n,d)
        
        ''' End of the definition of model_selection class'''
        
        self.model_selection = model_selection(n,d)
        





    def solve(self):
        
        Lformulation = ['LS','Concomitant','Concomitant_Huber','Huber']
        
        data = self.data
        
        X, C, y = data.X, data.C , data.y
        
        n,d      = len(y), len(X[0])
        matrices = (X,C,y)
        rho      = self.rho
        
        ''' define the class classo_data that contains the solution '''
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

        ''' End of the definition''' 
        
        
        solution = classo_solution()
        
        
        
        
        if data.rescale : 
            matrices, data.scaling = rescale(matrices)         #SCALING contains  :
                                                        #(list of initial norms of A-colomns, 
                                                        #         initial norm of centered y,
                                                        #          mean of initial y )
        model_selection = self.model_selection
        
        if model_selection.CV : 
            
            t0 = time()
            param = model_selection.CVparameters
            
            
            ''' Formulation choosing '''
            if param.formulation == 'not specified' : param.formulation = self.formulation
            formulation           = param.formulation
            if not formulation in Lformulation:
                print("One of the formulations used does not exist yet on classo, please try one of the following : \n  -LS \n -Concomitant \n -Concomitant_Huber \n -Huber")
                return()
            
           
            K                     = param.Nsubset
            lamin                 = param.lamin
            seed                  = param.seed
            
            numerical_method = choose_numerical_method(param.numerical_method, 'CV', formulation)
            param.numerical_method = numerical_method
            
            
            output = CV(matrices,K,typ=formulation,meth=numerical_method,lamin=lamin,seed=seed)
            
            if len(output) == 2 :    solution.beta_CV, solution.sigma_CV = output   
            else :                   solution.beta_CV = output
              
            solution.time_CV = time()-t0


        if model_selection.SS :
            
            t0 = time()
            param = model_selection.SSparameters
            
            ''' Formulation choosing '''
            if param.formulation == 'not specified' : param.formulation = self.formulation
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
                
  
            numerical_method = choose_numerical_method(param.numerical_method,'SS', formulation,
                                                       SSmethod = SSmethod, lam = lam)
            param.numerical_method = numerical_method

            solution.distribution_SS = stability(matrices, SSmethod = SSmethod, numerical_method = numerical_method,
                                    lam=lam, hd = hd, q = q ,B = B, pourcent_nS = pourcent_nS,
                                    formulation = formulation,plot_time=False,seed=seed)
            

            solution.time_SS = time()-t0
            
            
        if model_selection.LAMfixed : 
        
            t0 = time()
            param = model_selection.LAMparameters
            
            ''' Formulation choosing '''
            if param.formulation == 'not specified' : param.formulation = self.formulation
            formulation       = param.formulation
            if not formulation in Lformulation:
                print("One of the formulations used does not exist yet on classo, please try one of the following : \n  -LS \n -Concomitant \n -Concomitant_Huber \n -Huber")
                return()
            
            
            
            numerical_method  = param.formulation
            
            
            if param.lam == 'theoritical' : lam = param.theoritical_lam
            else                          : lam = param.lam 
            
            numerical_method = choose_numerical_method(param.numerical_method,'LAM', formulation, lam = lam)
            param.numerical_method = numerical_method
            
            
            output = Classo(matrices,lam,
                            typ = formulation, meth=numerical_method, 
                            plot_time=False , plot_sol=False, plot_sigm=False , rho = rho,get_lambdamax = True)
            
            if len(output) == 3 : solution.lambdamax, solution.beta_LAMfixed, solution.sigma_LAMfixed = output  
            else                : solution.lambdamax, solution.beta_LAMfixed = output
                
            solution.time_LAMfixed = time()-t0

        self.solution = solution
        
        

 




''' Annex function in order to choose the right numerical method, if the one gave is invalid''' 
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
                    

          
            
                    
                    
                    
                    
                    
                
                

