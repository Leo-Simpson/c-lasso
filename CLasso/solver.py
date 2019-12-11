from time import time
import numpy as np
import matplotlib.pyplot as plt

from CLasso.little_functions import random_data, csv_to_mat, rescale, theoritical_lam, min_LS
import scipy.io as sio
from CLasso.compact_func import Classo, pathlasso
from CLasso.cross_validation import CV
from CLasso.stability_selection import stability, selected_param
import matplotlib.patches as mpatches


'''
We build a class called classo_problem, that will contains all the information about the problem, settled in 4 categories : 
       - data : the matrices X, C, y to solve a problem of type y = X beta + sigma.epsilon under the constraint C.beta = 0
        
       - problem formulation : to know the formulation of the problem, huber function or not ; jointly estimate of sigma or not..
       
       - model selection : Cross Validation ; stability selection ; or Lasso problem for a fixed lambda. also contains the parameters of each of those model selection
       
       - solution : once we used the method .solve() , this componant will be added, with the solutions of the model-selections selected, with respect to the problem formulation selected


'''
        
class classo_problem :
    
    def __init__(self,X,y,C='zero-sum'): #zero sum constraint by default, but it can be any matrix

        n,d = len(X),len(X[0])

        
        ''' define the class classo_data that contains the data '''
        class classo_data : 
            def __init__(self,X,y,C):
                self.rescale = False               # booleen to know if we rescale the matrices
                self.X = X
                self.y = y
                if type(C)==str : C = np.ones( (1,len(X[0])) )
                self.C = C
        ''' End of the definition''' 
        
        self.data = classo_data(X,y,C)
        
        
        
        class classo_formulation :
            def __init__(self):
                self.huber = False
                self.concomitant = True
                self.rho = 1.345
                
            def name(self):
                if self.concomitant : 
                    if self.huber : return('Concomitant_Huber')
                    else     : return('Concomitant')
                else :
                    if self.huber : return('Huber')
                    else     : return('LS')
            def __repr__(self): return(self.name())
        

        self.formulation      = classo_formulation()
        '''
        define the class model_selection inside the class classo_problem
        '''
        class model_selection :
            def __init__(self,n,d):
                
                # Model selection parameters

                ''' CROSS VALIDATION PARAMETERS'''
                self.CV = False                          
                class CVparameters : 
                    def __init__(self):

                        self.seed             = 1
                        self.formulation      = 'not specified'     
                        self.numerical_method = 'choose'            
                        # can be : '2prox' ; 'ODE' ; 'Noproj' ; 'FB' ; and any other will make the algorithm decide

                        self.Nsubset          = 5                       # Number of subsets used
                        self.lamin            =  1e-2
                    def __repr__(self): return('Nsubset = '+str(self.Nsubset) 
                                               + '  lamin = '+ str(self.lamin)
                                               + ';  numerical_method = '+ str(self.numerical_method))
                ''' End of the definition''' 
                
                self.CVparameters = CVparameters()

                
                
                ''' STABILITY SELECTION PARAMETERS'''
                self.SS = True 
                class SSparameters :
                    def __init__(self,n,d):
                        self.seed = 1
                        self.formulation      = 'not specified'      
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
                        self.threshold        = 0.9
                        
                    def __repr__(self): return('method = '+str(self.method) 
                                               + ';  lamin = '+ str(self.lamin)
                                               + ';  B = '+ str(self.B)
                                               + ';  q = '+ str(self.q)
                                               + ';  pourcent_nS = '+ str(self.pourcent_nS)
                                               + ';  threshold = '+ str(self.threshold)
                                               + ';  numerical_method = '+ str(self.numerical_method))
                ''' End of the definition''' 
                
                self.SSparameters = SSparameters(n,d)


                
                
                
                ''' PROBLEM AT A FIXED LAMBDA PARAMETERS'''
                self.LAMfixed = False             
                class LAMfixedparameters : 
                    def __init__(self,n,d):
                        self.lam              = 'theoritical'
                        self.theoritical_lam  = round(theoritical_lam(n,d),3)
                        self.formulation      = 'not specified'      
                        self.numerical_method = 'choose'            
                        # can be : '2prox' ; 'ODE' ; 'Noproj' ; 'FB' ; and any other will make the algorithm decide
                    def __repr__(self): return('lam = '+str(self.lam) 
                                               + ';  theoritical_lam = '+ str(self.theoritical_lam)
                                               + ';  numerical_method = '+ str(self.numerical_method))
                ''' End of the definition''' 
                
                self.LAMfixedparameters = LAMfixedparameters(n,d)
                
            def __repr__(self) : 
                string = ''
                if self.CV : string+='CV,  '
                if self.SS : string+='SS, '
                if self.LAMfixed : string+= ' LAMfixed'
                return string
        ''' End of the definition of model_selection class'''
        
        self.model_selection = model_selection(n,d)
        





    def solve(self):
        
        data = self.data
        matrices = (data.X, data.C , data.y)
        solution = classo_solution()
        
        if data.rescale : 
            matrices, data.scaling = rescale(matrices)         #SCALING contains  :
                                                        #(list of initial norms of A-colomns, 
                                                        #         initial norm of centered y,
                                                        #          mean of initial y )
        

        #Compute the cross validation thanks to the class solution_CV which contains directely the computation in the initialisation
        if self.model_selection.CV : 
            solution.CV = solution_CV(matrices,self.model_selection.CVparameters,self.formulation)
            
            
        #Compute the Stability Selection thanks to the class solution_SS which contains directely the computation in the initialisation
        if self.model_selection.SS : 
            solution.SS = solution_SS(matrices,self.model_selection.SSparameters,self.formulation)
            
            
        #Compute the c-lasso problem at a fixed lam thanks to the class solution_LAMfixed which contains directely the computation in the initialisation
        if self.model_selection.LAMfixed : 
            solution.LAMfixed = solution_LAMfixed(matrices,self.model_selection.LAMfixedparameters,self.formulation)
        
        self.solution=solution

        
        




''' define the class classo_data that contains the solution '''
class classo_solution : 
    
    def __init__(self) : 
        self.CV       = 'not computed'
        self.SS       = 'not computed'
        self.LAMfixed = 'not computed'

    def __repr__(self):
        return(         'Cross Validation         : ' + self.CV.__repr__() 
               + '\n' + 'Stability Selection      : ' + self.SS.__repr__() 
               + '\n' + 'Solution for a fixed lam : ' + self.LAMfixed.__repr__() )
                    
                    
        
        
        
        
class solution_CV:
    
    def __init__(self,matrices,param,formulation):
        t0 = time()

        #Formulation choosing
        if param.formulation == 'not specified' : param.formulation = formulation
        name_formulation       = param.formulation.name()
        
        rho = param.formulation.rho
        # Algorithmic method choosing
        numerical_method = choose_numerical_method(param.numerical_method, 'CV', param.formulation)
        param.numerical_method = numerical_method

        # Compute the solution and is the formulation is concomitant, it also compute sigma
        out, LAM,i,AVG,SE     =  CV(matrices,param.Nsubset,
                                    typ=name_formulation,meth=numerical_method,
                                    lamin=param.lamin,seed=param.seed,rho=rho)

        if param.formulation.concomitant : self.beta, self.sigma = out
        else : self.beta                    = out

        self.selected_param = self.beta != 0. # boolean array, false iff beta_i =0
        self.refit = min_LS(matrices,self.selected_param)
        self.time = time()-t0  
        
    def __repr__(self):
        plt.bar(range(len(self.refit)),self.refit),   plt.title("Cross Validation refit"),   plt.show()
        return (    " Running time for Cross Validation    : "  + str(round(self.time,3))       +"s")
        
                    
class solution_SS:
    def __init__(self,matrices,param,formulation):
        t0 = time()

        #Formulation choosing
        if param.formulation == 'not specified' : param.formulation = formulation
        name_formulation       = param.formulation.name()

        rho = param.formulation.rho
        #Compute the theoritical lam if necessary
        if param.lam == 'theoritical' : lam = param.theoritical_lam
        else                          : lam = param.lam 


        # Algorithmic method choosing
        numerical_method = choose_numerical_method(param.numerical_method,'SS', param.formulation,
                                                   SSmethod = param.method, lam = lam)
        param.numerical_method = numerical_method  


        # Compute the distribution
        output = stability(matrices, SSmethod = param.method, numerical_method = numerical_method,
                                lam=lam, hd = param.hd, q = param.q ,B = param.B, pourcent_nS = param.pourcent_nS,
                                formulation = name_formulation,plot_time=False,seed=param.seed,rho=rho)

        if (param.method =='first'): distribution, distribution_path,lambdas = output
        else                       : distribution, distribution_path,lambdas = output,'not computed','not used'

        self.distribution      = distribution
        self.distribution_path = distribution_path
        self.lambdas_path         = lambdas
        self.selected_param = selected_param(self.distribution,param.threshold)

        self.refit = min_LS(matrices,self.selected_param)
        self.time = time()-t0            
    
    def __repr__(self):
        
        D                      = self.distribution
        Dpath                  = self.distribution_path
        selected               = self.selected_param
        unselected             = [not i for i in selected]
        Dselected              = np.zeros(len(D))
        Dunselected            = np.zeros(len(D))
        Dselected[selected]    = D[selected]
        Dunselected[unselected]= D[unselected]

        plt.bar(range(len(Dselected)),Dselected,color = 'r' ,label = 'selected parameters')
        plt.bar(range(len(Dunselected)),Dunselected,color = 'b',label = 'unselected parameters')
        plt.legend(),plt.title("Distribution of Stability Selection"), plt.show()

        if (type(Dpath)!= str): 
            lambdas = self.lambdas_path
            N = len(lambdas)
            for i in range(len(selected)):
                if selected[i] :plt.plot(lambdas,[Dpath[j][i] for j in range(N)],'r',label = 'selected parameters')
                else           :plt.plot(lambdas,[Dpath[j][i] for j in range(N)],'b',label = 'unselected parameters')
            p1,p2 = mpatches.Patch(color='red', label='selected parameters'),mpatches.Patch(color='blue', label='unselected parameters')

            plt.legend(handles=[p1,p2])
            plt.title("Distribution of probability of apparence as a function of lambda"),plt.show()
        
        
        
        
        plt.bar(range(len(self.refit)),self.refit),   plt.title("Solution for Stability Selection with refit"),   plt.show()
        return (    " Running time for Stability Selection    : "  + str(round(self.time,3))       +"s")
        
        
        
class solution_LAMfixed :
    def __init__(self,matrices,param,formulation):
        t0 = time()

        #Formulation choosing
        if param.formulation == 'not specified' : param.formulation = formulation
        name_formulation       = param.formulation.name()

        rho = param.formulation.rho
        #Compute the theoritical lam if necessary
        if param.lam == 'theoritical' : lam = param.theoritical_lam
        else                          : lam = param.lam 


        # Algorithmic method choosing
        numerical_method = choose_numerical_method(param.numerical_method,'LAM', param.formulation, lam = lam)
        param.numerical_method = numerical_method

        # Compute the solution and is the formulation is concomitant, it also compute sigma
        out = Classo(
                matrices,lam, typ = name_formulation, meth=numerical_method, 
                plot_time=False , plot_sol=False, plot_sigm=False , rho = rho,get_lambdamax = True)
        
        if param.formulation.concomitant : 
            self.lambdamax, self.beta, self.sigma = out
        else : self.lambdamax, self.beta          = out


        self.selected_param = self.beta !=0.
        self.refit = min_LS(matrices,self.selected_param)
        self.time = time()-t0
        
    def __repr__(self):
        plt.bar(range(len(self.refit)),self.refit),   plt.title("Solution for a fixed lambda with refit"),   plt.show()
        return (    " Running time for LAM fixed       : "  + str(round(self.time,3))       +"s")

        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
                

''' Annex function in order to choose the right numerical method, if the one gave is invalid''' 
def choose_numerical_method(method,model,formulation,SSmethod = None, lam = None):
    
    if (formulation.concomitant and formulation.huber):
        if not method in ['2prox']: return '2prox'
    
    
        
    # cases where we use classo at a fixed lambda    
    elif (model == 'LAM') or (model == 'SS' and SSmethod == 'lam') : 
              
        
        
        if formulation.concomitant :
            if not method in ['ODE','2prox']:
                if (lam>0.1): return 'ODE'
                else        : return '2prox'
      
        else:
            if not method in ['ODE','2prox','FB','Noproj']:
                if (lam>0.1): return 'ODE'
                else        : return '2prox'     
    
    
    
    # cases where we use pathlasso                
    else:
        if formulation.concomitant :
            if not method in ['ODE','2prox']: return 'ODE'
            
        else:
            if not method in ['ODE','2prox','FB','Noproj']: return 'ODE'         
            
    return method
