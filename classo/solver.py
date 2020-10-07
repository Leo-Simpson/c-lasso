from time import time
import numpy as np
import matplotlib.pyplot as plt

from .misc_functions import rescale, theoretical_lam, min_LS, affichage, check_size, tree_to_matrix
from .compact_func import Classo, pathlasso
from .cross_validation import CV
from .stability_selection import stability, selected_param
import matplotlib.patches as mpatches


class classo_problem:
    ''' Class that contains all the information about the problem
        
        Args:
        X (ndarray): Matrix representing the data of the problem
        y (ndarray): Vector representing the output of the problem
        C (str or ndarray, optional ): Matrix of constraints to the problem. If it is 'zero-sum' then the corresponding attribute will be all-one matrix.
        Default value to 'zero-sum'
        rescale (bool, optional): if True, then the function :func:`rescale` will be applied to data when solving the problem.
        Default value is 'False'
        label (list) : list of the labels of each variable. If None, then label or just int
        
        Attributes:
        data (classo_data) :  object containing the data of the problem.
        formulation (classo_formulation) : object containing the info about the formulation of the minimization problem we solve.
        model_selection (classo_model_selection) : object giving the parameters we need to do variable selection.
        solution (classo_solution) : object giving caracteristics of the solution of the model_selection that is asked.
            Before using the method solve() , its componant are empty/null.
        numerical_method (str) : name of the numerical method that is used, it can be :
            'Path-Alg' (path algorithm) , 'P-PDS' (Projected primal-dual splitting method) , 'PF-PDS' (Projection-free primal-dual splitting method) or 'DR' (Douglas-Rachford-type splitting method)
            Default value : 'not specified', which means that the function :func:`choose_numerical_method` will choose it accordingly to the formulation
        
        '''
    def __init__(self, X, y, C=None,Tree = None, label=None, rescale=False):  # zero sum constraint by default, but it can be any matrix
        self.data = classo_data(X, y, C,Tree=Tree, rescale=rescale, label = label)
        self.formulation = classo_formulation()
        self.model_selection = classo_model_selection()
        self.solution = classo_solution()
        self.numerical_method = "not specified"
    
    
    # This method is the way to solve the model selections contained in the object model_selection, with the formulation of 'formulation' and the data.
    def solve(self):
        ''' Method that solve every model required in the attributes of the problem and update the attribute :obj:`problem.solution` with the characteristics of the solution.
            '''
        data = self.data
        matrices = (data.X, data.C, data.y)
        solution = classo_solution()
        n, d = len(data.X), len(data.X[0])
        if self.formulation.classification :
            self.formulation.concomitant = False
        
        if(type(self.formulation.e) == str):
            if (self.formulation.e == 'n/2'): self.formulation.e = n/2  #useful to be able to write e='n/2' as it is in the default parameters
            elif(self.formulation.e == 'n'): self.formulation.e = n     # same
            else :
                if (self.formulation.huber): self.formulation.e = n
                else                       : self.formulation.e = n / 2
        
        if data.rescale:
            matrices, data.scaling = rescale(matrices)  # SCALING contains  :
                                                                # (list of initial norms of A-colomns,
                                                                #         initial norm of centered y,
                                                                #          mean of initial y )

        if not self.formulation.w is None : 
            if min(self.formulation.w) < 1e-3 : 
                raise ValueError("w has to be positive weights, here it has a value smaller than 1e-3")

        if self.formulation.intercept : 
            data.label = np.array(['intercept']+list(data.label))
        
        label = data.label

        # Compute the path thanks to the class solution_path which contains directely the computation in the initialisation
        if self.model_selection.PATH:
            solution.PATH = solution_PATH(matrices, self.model_selection.PATHparameters, self.formulation, self.numerical_method, label)
        
        # Compute the cross validation thanks to the class solution_CV which contains directely the computation in the initialisation
        if self.model_selection.CV:
            solution.CV = solution_CV(matrices, self.model_selection.CVparameters, self.formulation, self.numerical_method, label)
    
        # Compute the Stability Selection thanks to the class solution_SS which contains directely the computation in the initialisation
        if self.model_selection.StabSel:
            param = self.model_selection.StabSelparameters
            param.theoretical_lam = theoretical_lam(int(n * param.percent_nS), d)
            if not param.rescaled_lam : param.theoretical_lam = param.theoretical_lam*int(n * param.percent_nS)
            
            solution.StabSel = solution_StabSel(matrices, param, self.formulation, self.numerical_method, label)
        
        # Compute the c-lasso problem at a fixed lam thanks to the class solution_LAMfixed which contains directely the computation in the initialisation
        if self.model_selection.LAMfixed:
            param = self.model_selection.LAMfixedparameters
            param.theoretical_lam = theoretical_lam(n, d)
            if not param.rescaled_lam: param.theoretical_lam = param.theoretical_lam*n
            solution.LAMfixed = solution_LAMfixed(matrices, param, self.formulation, self.numerical_method, label)

        self.solution = solution
    
    def __repr__(self):
        print_parameters = ''
        if (self.model_selection.LAMfixed):
            print_parameters += '\n \nLAMBDA FIXED PARAMETERS: ' + self.model_selection.LAMfixedparameters.__repr__()
        
        if (self.model_selection.PATH):
            print_parameters += '\n \nPATH PARAMETERS: ' + self.model_selection.PATHparameters.__repr__()
        
        if (self.model_selection.CV):
            print_parameters += '\n \nCROSS VALIDATION PARAMETERS: ' + self.model_selection.CVparameters.__repr__()
        
        if (self.model_selection.StabSel):
            print_parameters += '\n \nSTABILITY SELECTION PARAMETERS: ' + self.model_selection.StabSelparameters.__repr__()

        return (' \n \nFORMULATION: ' + self.formulation.__repr__()
                + '\n \n' +
                'MODEL SELECTION COMPUTED:  ' + self.model_selection.__repr__()
                + print_parameters + '\n'
                )




class classo_data:
    ''' Class containing the data of the problem

    Args:
        X (ndarray): Matrix representing the data of the problem
        y (ndarray): Vector representing the output of the problem
        C (str or array, optional ): Matrix of constraints to the problem. If it is 'zero-sum' then the corresponding attribute will be all-one matrix.
        rescale (bool, optional): if True, then the function :func:`rescale` will be applied to data when solving the problem
        label (list) : list of the labels of each variable. If None, then label or just int
        Tree (skbio.TreeNode) : taxonomic tree, if set it is not None, then the matrices X and C and the labels will be changed. 

    Attributes:
        X (ndarray): Matrix representing the data of the problem
        y (ndarray): Vector representing the output of the problem
        C (str or array, optional ): Matrix of constraints to the problem. If it is 'zero-sum' then the corresponding attribute will be all-one matrix.
        rescale (bool, optional): if True, then the function :func:`rescale` will be applied to data when solving the problem
        label (list) : list of the labels of each variable. If None, then label or just int
        tree (skbio.TreeNode) : taxonomic tree 

    '''

    def __init__(self, X, y, C, Tree=None, rescale=False, label=None):
        self.rescale = rescale  # booleen to know if we rescale the matrices
        X1,y1,C1 = check_size(X,y,C)

        if Tree is None : 
            if (label is None): self.label = np.array([str(i) for i in range(len(X[0]))])
            else : self.label = np.array(label)
            self.X,self.y,self.C, self.tree = X1,y1,C1, None

        else : 
            A, label2, subtree = tree_to_matrix(Tree,label, with_repr=True)
            self.tree = subtree
            self.X,self.y,self.C, self.label = X1.dot(A),y1,C1.dot(A), label2
            
       
class classo_formulation:
    ''' Class containing the data of the problem

    Attributes:
        huber (bool) : True if the formulation of the problem should be robust
            Default value = False

        concomitant (bool) : True if the formulation of the problem should be with an M-estimation of sigma.
            Default value = True

        classification (bool) : True if the formulation of the problem should be classification (if yes, then it will not be concomitant)
            Default value = False

        rho (float) : Value of rho for robust problem.
            Default value = 1.345

        rho_classification (float) : value of rho for huberized hinge loss function for classification (this parameter has to be negative).
            Default value = -1.

        e (float or string)  : value of e in concomitant formulation.
            If 'n/2' then it becomes n/2 during the method solve(), same for 'n'.
            Default value : 'n' if huber formulation ; 'n/2' else

        w (numpy ndarray) : array of size d with the weights of the L1 penalization
            Default value : None (which makes it the 1,...,1 vector)

        intercept (bool)  : set to true if we should use an intercept
            Default value : False

    '''

    def __init__(self):
        self.huber = False
        self.concomitant = True
        self.classification = False
        self.rho = 1.345
        self.rho_classification = -1.
        self.e = 'not specified'
        self.w = None
        self.intercept = False

    def name(self):
        if self.classification:
            if self.huber:
                return "C2"
            else : return "C1"
        if self.concomitant:
            if self.huber:
                return "R4"
            else : return "R3"
        if self.huber:
            return "R2"
        else : return "R1"

    def __repr__(self):
        return (self.name())

class classo_model_selection:
    ''' Class containing the data of the problem

    Attributes:
        PATH (bool): True if path should be computed.
            Default Value = False

        PATHparameters (PATHparameters): object parameters to compute the lasso-path.


        CV (bool):  True if Cross Validation should be computed.
            Default Value = False

        CVparameters (CVparameters):  object parameters to compute the cross-validation.


        StabSel (boolean):  True if Stability Selection should be computed.
            Default Value = True

        StabSelparameters (StabSelparameters):  object parameters to compute the stability selection.

        LAMfixed (boolean):  True if solution for a fixed lambda should be computed.
            Default Value = False

        LAMfixedparameters (LAMparameters):  object parameters to compute the lasso for a fixed lambda

    '''
    def __init__(self, method = "not specified"):

        # Model selection variables

        self.PATH = False
        self.PATHparameters = PATHparameters(method=method) 

        self.CV = False
        self.CVparameters = CVparameters(method=method)

        self.StabSel = True            # Only model selection that is used by default
        self.StabSelparameters = StabSelparameters(method=method)

        self.LAMfixed = False
        self.LAMfixedparameters = LAMfixedparameters(method=method)

    def __repr__(self):
        string = ''
        if self.LAMfixed: string += '\n     Lambda fixed'
        if self.PATH: string +=     '\n     Path'
        if self.CV: string +=       '\n     Cross Validation'
        if self.StabSel: string +=  '\n     Stability selection'
        
        return string

class PATHparameters:
    '''object parameters to compute the lasso-path.
    
    Attributes:
        numerical_method (str) : name of the numerical method that is used, it can be :
            'Path-Alg' (path algorithm) , 'P-PDS' (Projected primal-dual splitting method) , 'PF-PDS' (Projection-free primal-dual splitting method) or 'DR' (Douglas-Rachford-type splitting method)
            Default value : 'not specified', which means that the function :func:`choose_numerical_method` will choose it accordingly to the formulation

        n_active (int): if it is higher than 0, then the algo stop computing the path when n_active variables are actives. then the solution does not change from this point.
            Dafault value : 0

        lambdas (numpy.ndarray) : list of lambdas for computinf lasso-path for cross validation on lambda.
            Default value : np.array([10**(-delta * float(i) / Nlam) for i in range(0,Nlam) ] ) with delta=2. and Nlam = 40

        plot_sigma (bool) : if True then the print method of the solution will also show sigma if it is computed (formulation R3 or R4)
            Default value : True

        label (numpy.ndarray of str) : labels on each coefficients
    
    '''
    def __init__(self, method="not specified"):
        self.formulation = 'not specified'
        self.numerical_method = method
        self.n_active = 0
        lamin= 1e-2
        Nlam = 40
        self.lambdas = np.array([10**(np.log10(lamin) * float(i) / (Nlam+1)) for i in range(0,Nlam) ] )
        self.plot_sigma = True

    def __repr__(self): 
        string  = '\n     numerical_method : ' + str(self.numerical_method)
        string += '\n     Npath = ' + str(len(self.lambdas))
        string += '\n     lamin = ' + str(round(self.lambdas[-1],3))
        string += '\n     lamax = ' + str(round(self.lambdas[0],3))
        
        if self.n_active > 0 : 
            string += '\n     n_active = ' + str(self.n_active)

        return string
                                

class CVparameters:
    '''object parameters to compute the cross-validation.

    Attributes:
        seed (bool or int, optional) : Seed for random values, for an equal seed, the result will be the same. If set to False/None: pseudo-random seed
            Default value : 0

        numerical_method (str) : name of the numerical method that is used, can be :
            'Path-Alg' (path algorithm) , 'P-PDS' (Projected primal-dual splitting method) , 'PF-PDS' (Projection-free primal-dual splitting method) or 'DR' (Douglas-Rachford-type splitting method)
            Default value : 'not specified', which means that the function :func:`choose_numerical_method` will choose it accordingly to the formulation

        lambdas (numpy.ndarray) : list of lambdas for computinf lasso-path for cross validation on lambda.
            Default value : None 

        oneSE (bool) : if set to True, the selected lambda if computed with method 'one-standard-error'
            Default value : True

        Nsubset (int): number of subset in the cross validation method
            Dafault value : 5

    '''
    def __init__(self,method = "not specified"):
        self.seed = 0
        self.formulation = 'not specified'
        self.numerical_method = method

        self.Nsubset = 5  # Number of subsets used
        self.Nlam = 80
        self.lambdas = None
        self.oneSE = True

    def __repr__(self): 
        string  = '\n     numerical_method : ' + str(self.numerical_method)
        string += '\n     one-SE method : ' + str(self.oneSE)
        string += '\n     Nsubset = ' + str(self.Nsubset)
        string += '\n     lamin = ' + str(self.lambdas[-1])
        string += '\n     Nlam = ' + str(len(self.lambdas))
        
        
        return string

class StabSelparameters:
    '''object parameters to compute the stability selection.

    Attributes:

        seed (bool or int, optional) : Seed for random values, for an equal seed, the result will be the same. If set to False/None: pseudo-random seed
            Default value : 123

        numerical_method (str) : name of the numerical method that is used, can be :
            'Path-Alg' (path algorithm) , 'P-PDS' (Projected primal-dual splitting method) , 'PF-PDS' (Projection-free primal-dual splitting method) or 'DR' (Douglas-Rachford-type splitting method)
            Default value : 'not specified', which means that the function :func:`choose_numerical_method` will choose it accordingly to the formulation

        lam (float or str) : (only used if :obj:`method` = 'lam') lam for which the lasso should be computed.
            Default value : 'theoretical' which mean it will be equal to :obj:`theoretical_lam` once it is computed

        rescaled_lam (bool) : (only used if :obj:`method` = 'lam') False if lam = lambda, False if lam = lambda/lambdamax which is between 0 and 1.
            If False and lam = 'theoretical' , then it will takes the  value n*theoretical_lam.
            Default value : True


        theoretical_lam (float) : (only used if :obj:`method` = 'lam') Theoretical lam.
            Default value : 0.0 (once it is not computed yet, it is computed thanks to the function :func:`theoretical_lam` used in :meth:`classo_problem.solve`)


        method (str) : 'first', 'lam' or 'max' depending on the type of stability selection we do.
            Default value : 'first'

        B (int) : number of subsample considered.
            Default value : 50

        q (int) : number of selected variable per subsample.
            Default value : 10

        percent_nS (float) : size of subsample relatively to the total amount of sample
            Default value : 0.5

        lamin (float) : lamin when computing the lasso-path for method 'max'
            Default value : 1e-2

        hd (bool) : if set to True, then the 'max' will stop when it reaches n-k actives variables
            Default value : False

        threshold (float) : threhold for stability selection
            Default value : 0.7

        threshold_label (float) : threshold to know when the label should be plot on the graph.
            Default value : 0.4

    '''
    def __init__(self, method="not specified"):
        self.seed = 123
        self.formulation = 'not specified'
        self.numerical_method = method

        self.method = 'first'  # Can be 'first' ; 'max' or 'lam'
        self.B = 50
        self.q = 10
        self.percent_nS = 0.5
        self.Nlam = 50      # for path computation
        self.lamin = 1e-2   # the lambda where one stop for 'max' method
        self.hd = False     # if set to True, then the 'max' will stop when it reaches n-k actives variables
        self.lam = 'theoretical'  # can also be a float, for the 'lam' method
        self.rescaled_lam = True
        self.threshold = 0.7
        self.threshold_label = 0.4
        self.theoretical_lam = 0.0

    def __repr__(self): 
        string  = '\n     numerical_method : ' + str(self.numerical_method)
        string += '\n     method : ' + str(self.method)
        string += '\n     B = ' + str(self.B)
        string += '\n     q = ' + str(self.q)
        string += '\n     percent_nS = ' + str(self.percent_nS)
        string += '\n     threshold = ' + str(self.threshold)
        

        if self.method == 'lam': 
            string += '\n     lam = ' + str(self.lam)
            if self.theoretical_lam != 0.  :
                string += '\n     theoretical_lam = ' + str(round(self.theoretical_lam, 4))
        else : 
            string += '\n     lamin = ' + str(self.lamin) 
            string += '\n     Nlam = '+ str(self.Nlam)

        return string

class LAMfixedparameters:
    '''object parameters to compute the lasso for a fixed lambda

    Attributes:
        numerical_method (str) : name of the numerical method that is used, can be :
            'Path-Alg' (path algorithm) , 'P-PDS' (Projected primal-dual splitting method) , 'PF-PDS' (Projection-free primal-dual splitting method) or 'DR' (Douglas-Rachford-type splitting method)
            Default value : 'not specified', which means that the function :func:`choose_numerical_method` will choose it accordingly to the formulation

        lam (float or str) : lam for which the lasso should be computed.
            Default value : 'theoretical' which mean it will be equal to :obj:`theoretical_lam` once it is computed

        rescaled_lam (bool) : False if lam = lambda, True if lam = lambda/lambdamax which is between 0 and 1.
            If False and lam = 'theoretical' , then it will takes the  value n*theoretical_lam.
            Default value : True

        theoretical_lam (float) : Theoretical lam
            Default value : 0.0 (once it is not computed yet, it is computed thanks to the function :func:`theoretical_lam` used in :meth:`classo_problem.solve`)

        threshold (float) : Threshold such that the parameters i selected or the ones such as | beta_i | > threshold
            If None, then it will be set to the average of the vector |beta|
            Default value : None
    '''
    def __init__(self, method="not specified"):
        self.lam = 'theoretical'
        self.formulation = 'not specified'
        self.numerical_method = method
        self.rescaled_lam = True
        self.theoretical_lam = 0.0
        self.threshold = None

    def __repr__(self): 
        string  = '\n     numerical_method = ' + str(self.numerical_method)
        string += '\n     rescaled lam : ' + str(self.rescaled_lam)
        string += '\n     threshold = ' + str(round(self.threshold,3))
        if type(lam) is str : string += '\n     lam : ' + lam
        else : string += '\n     lam = ' + str(round(self.lam,3))
        
        
        if self.theoretical_lam != 0.  :
            string += '\n     theoretical_lam = ' + str(round(self.theoretical_lam, 4))
        
        return string

class classo_solution:
    ''' Class giving characteristics of the solution of the model_selection that is asked.
                                      Before using the method solve() , its componant are empty/null.


    Attributes:
        PATH (solution_PATH): Solution components of the model PATH
        CV (solution_CV):  Solution components of the model CV
        StabelSel (solution_StabSel): Solution components of the model StabSel
        LAMfixed (solution_LAMfixed): Solution components of the model LAMfixed

    '''
    def __init__(self):
        self.PATH = 'not computed' #this will be filled with an object of the class 'solution_PATH' when the method solve() will be used.
        self.CV = 'not computed'  # will be an object of the class 'solution_PATH'
        self.StabSel = 'not computed' # will be an object of the class 'solution_StabSel'
        self.LAMfixed = 'not computed'

    def __repr__(self):
        string = ''
        if not type(self.LAMfixed) is str : 
            string  += self.LAMfixed.__repr__()  +  '\n'
        if not type(self.PATH) is str : 
            string  += self.PATH.__repr__()  +  '\n'
        if not type(self.CV) is str : 
            string  += self.CV.__repr__()  +  '\n'
        if not type(self.StabSel) is str : 
            string  += self.StabSel.__repr__()  +  '\n'



        return string


#Here, the main function used is pathlasso ; from the file compact_func
class solution_PATH:
    ''' Class giving characteristics of the lasso-path computed,
    which also contains a method _repr_ that plot the graphic of this lasso-path

    Attributes:
        BETAS (numpy.ndarray) : array of size Npath x d with the solution beta for each lambda on each row
        SIGMAS (numpy.ndarray) : array of size Npath with the solution sigma for each lambda when the formulation of the problem is R2 or R4
        LAMBDAS (numpy.ndarray) : array of size Npath with the lambdas (real lambdas, not divided by lambda_max) for which the solution is computed
        method (str) : name of the numerical method that has been used. It can be 'Path-Alg', 'P-PDS' , 'PF-PDS' or 'DR'
        save (bool or str) : if it is a str, then it gives the name of the file where the graphics has been/will be saved (after using print(solution) )
        formulation (str) : can be 'R1' ; 'R2' ; 'R3' ; 'R4' ; 'C1' ; 'C2'
        time (float) : running time of this action

    '''
    def __init__(self, matrices, param, formulation, numerical_method, label):
        t0 = time()

        # Formulation choosing
        if param.formulation == 'not specified': param.formulation = formulation
        if param.numerical_method == "not specified" : param.numerical_method = numerical_method
        name_formulation = param.formulation.name()
        rho = param.formulation.rho
        rho_classification = param.formulation.rho_classification
        e = param.formulation.e
        # Algorithmic method choosing
        numerical_method = choose_numerical_method(param.numerical_method, 'PATH', param.formulation)
        param.numerical_method = numerical_method
        # Compute the solution and is the formulation is concomitant, it also compute sigma

        out = pathlasso(matrices, lambdas=param.lambdas, n_active=param.n_active,
                                                typ=name_formulation, meth=numerical_method, return_sigm=True,
                                                rho=rho, e=e,rho_classification=rho_classification, w=param.formulation.w, 
                                                intercept = param.formulation.intercept)
        if(formulation.concomitant): self.BETAS, self.LAMBDAS, self.SIGMAS = out
        else :
            self.BETAS, self.LAMBDAS = out
            self.SIGMAS = 'not computed'

        self.formulation = formulation
        self.plot_sigma = param.plot_sigma
        self.method = numerical_method
        self.save = False
        self.label = label
        self.time = time() - t0

    def __repr__(self):

        string = "\n PATH COMPUTATION : "


        affichage(self.BETAS, self.LAMBDAS, labels=self.label, naffichage=5,
                  title=PATH_beta_path["title"] + self.formulation.name(),xlabel=PATH_beta_path["xlabel"],ylabel=PATH_beta_path["ylabel"])
        if (type(self.save) == str): plt.savefig(self.save + 'Beta-path')
        plt.show()
        if(type(self.SIGMAS)!=str and self.plot_sigma):
            plt.plot(self.LAMBDAS, self.SIGMAS), plt.ylabel(PATH_sigma_path["ylabel"]), plt.xlabel(PATH_sigma_path["xlabel"])
            plt.title(PATH_sigma_path["title"] + self.formulation.name())
            if (type(self.save)==str) : plt.savefig(self.save + 'Sigma-path')
            plt.show()

        string += "\n   Running time :  "  + str(round(self.time, 3)) + "s"
        return string

#Here, the main function used is CV ; from the file cross_validation
class solution_CV:
    ''' Class giving characteristics of the cross validation computed,
    which also contains a method _repr_() that plot the selected parameters and the solution of the not-sparse problem on the selected variables set
    It also contains a method gaphic(self, mse_max=1.,save=False) that computes the curve of validation error as a function of lambda

    Attributes:
        xGraph (numpy.ndarray) : array of size Nlam of the lambdas / lambda_max
        yGraph (numpy.ndarray) : array of size Nlam of the average validation residual (over the K subsets)
        standard_error (numpy.ndarray) : array of size Nlam of the standard error of the validation residual (over the K subsets)
        index_min (int) : index on xGraph of the selected lambda without 1-standard-error method
        index_1SE (int) : index on xGraph of the selected lambda with 1-standard-error method
        lambda_min (float) : selected lambda without 1-standard-error method
        lambda_oneSE (float) : selected lambda with 1-standard-error method
        beta (numpy.ndarray) : solution beta of classo at lambda_oneSE
        sigma (float) : solution sigma of classo at lambda_oneSE when formulation is 'R2' or 'R4'
        selected_param (numpy.ndarray) : boolean arrays of size d with True when the variable is selected
        refit (numpy.ndarray) : solution beta after solving unsparse problem over the set of selected variables.
        formulation (str) : can be 'R1' ; 'R2' ; 'R3' ; 'R4' ; 'C1' ; 'C2'
        time (float) : running time of this action

    '''
    def __init__(self, matrices, param, formulation, numerical_method, label):
        t0 = time()

        # Formulation choosing
        if param.formulation == 'not specified': param.formulation = formulation
        if param.numerical_method == "not specified" : param.numerical_method = numerical_method
        name_formulation = param.formulation.name()

        rho = param.formulation.rho
        rho_classification = param.formulation.rho_classification
        e = param.formulation.e
        # Algorithmic method choosing
        numerical_method = choose_numerical_method(param.numerical_method, 'CV', param.formulation)
        param.numerical_method = numerical_method


        if param.lambdas is None : param.lambdas = np.linspace(1.,1e-3,param.Nlam)

        # Compute the solution and is the formulation is concomitant, it also compute sigma
        out, self.yGraph, self.standard_error, self.index_min, self.index_1SE = CV(matrices, param.Nsubset,
                                                                                   typ=name_formulation,
                                                                                   num_meth=numerical_method,
                                                                                   lambdas=param.lambdas,
                                                                                   seed=param.seed, rho=rho,
                                                                                   rho_classification=rho_classification,
                                                                                   oneSE=param.oneSE, e=e, w=param.formulation.w,
                                                                                   intercept = param.formulation.intercept)

        self.xGraph = param.lambdas
        self.lambda_1SE = param.lambdas[self.index_1SE]
        self.lambda_min = param.lambdas[self.index_min]

        if param.formulation.concomitant:
            self.beta, self.sigma = out
        else:
            self.beta = out

        self.selected_param = abs(self.beta) > 1e-3  # boolean array, false iff beta_i =0
        self.refit = min_LS(matrices, self.selected_param, intercept=param.formulation.intercept)
        self.time = time() - t0
        self.save=False
        self.label = label

    def __repr__(self):

        string = "\n CROSS VALIDATION : "

        plt.bar(range(len(self.refit)), self.refit), plt.title(CV_beta["title"]), plt.xlabel(CV_beta["xlabel"]),plt.ylabel(CV_beta["ylabel"])
        plt.xticks(np.where(self.selected_param)[0],self.label[self.selected_param], rotation=30)
        if(type(self.save)==str): plt.savefig(self.save)
        plt.show()
        
        string += "\n   Selected variables :  " 
        for i in np.where(self.selected_param)[0] :
            string += self.label[i] + "    "

        string += "\n   Running time :  "  + str(round(self.time, 3)) + "s"
        return string

    def graphic(self, ratio_mse_max=1.,save=False):
        i_min, i_1SE = self.index_min, self.index_1SE
        mse_max = ratio_mse_max * self.standard_error[i_min]
        j = 0
        while(j < i_1SE - 30  and self.yGraph[j] > mse_max) : j+=1

        plt.errorbar(self.xGraph[j:], self.yGraph[j:], self.standard_error[j:], label='mean over the k groups of data', errorevery = 10 )
        plt.axvline(x=self.xGraph[i_min], color='k', label=r'$\lambda$ (min MSE)')
        plt.axvline(x=self.xGraph[i_1SE],color='r',label=r'$\lambda$ (1SE) ')
        plt.title(CV_graph["title"]), plt.xlabel(CV_graph["xlabel"]),plt.ylabel(CV_graph["ylabel"])
        plt.legend()
        if(type(save)==str) : plt.savefig(save)
        plt.show()

#Here, the main function used is stability ; from the file stability selection
class solution_StabSel:
    ''' Class giving characteristics of the stability selection computed,
    which also contains a method _repr_() that plot the selected parameters,
    the solution of the not-sparse problem on the selected variables set, the stability plot with the evolution of it with lambda if the used method is 'first'

    Attributes:
        distribution (array) : d array of stability rations.
        lambdas_path (array or string) : for 'first' method : Nlam array of the lambdas used. Other cases : 'not used'
        distribution_path (array or string) : for 'first' method :  Nlam x d array with stability ratios as a function of lambda. Other cases : 'not computed'
        threshold (float) : threshold for StabSel, ie for a variable i, stability ratio that is needed to get selected
        save1,save2,save3 (bool or string) : if a string is given, the corresponding graph will be saved with the given name of the file (save1 is for stability plot ; save2 for path-stability plot; and save3 for refit beta-solution)
        selected_param (numpy.ndarray) : boolean arrays of size d with True when the variable is selected
        to_label (numpy.ndarray) : boolean arrays of size d with True when the name of the variable should be seen on the graph
        refit (numpy.ndarray) : solution beta after solving unsparse problem over the set of selected variables.
        formulation (str) : can be 'R1' ; 'R2' ; 'R3' ; 'R4' ; 'C1' ; 'C2'
        time (float) : running time of this action

    '''
    def __init__(self, matrices, param, formulation, numerical_method, label):
        t0 = time()

        # Formulation choosing
        if param.formulation == 'not specified': param.formulation = formulation
        if param.numerical_method == "not specified" : param.numerical_method = numerical_method
        name_formulation = param.formulation.name()

        rho = param.formulation.rho
        rho_classification = param.formulation.rho_classification
        e = param.formulation.e
        # Compute the theoretical lam if necessary
        if param.lam == 'theoretical':
            lam = param.theoretical_lam
        else:
            lam = param.lam

        # Algorithmic method choosing
        numerical_method = choose_numerical_method(param.numerical_method, 'StabSel', param.formulation,
                                                   StabSelmethod=param.method, lam=lam)
        param.numerical_method = numerical_method

        # verify the method 
        if not param.method in ['first','max','lam']:
            raise ValueError("name of the stability selection method should be one of those : 'first' , 'max' , 'lam'     not {}".format(param.method))

        # Compute the distribution
        output = stability(matrices, StabSelmethod=param.method, numerical_method=numerical_method, lamin=param.lamin,
                           lam=lam,Nlam=param.Nlam, q=param.q, B=param.B, percent_nS=param.percent_nS,
                           formulation=name_formulation, seed=param.seed, rho=rho,
                           rho_classification=rho_classification,
                           true_lam= not param.rescaled_lam, e=e, w=param.formulation.w, intercept = param.formulation.intercept)

        if (param.method == 'first'):
            distribution, distribution_path, lambdas = output
        else:
            distribution, distribution_path, lambdas = output, 'not computed', 'not used'

        self.distribution = distribution
        self.distribution_path = distribution_path
        self.lambdas_path = lambdas
        self.selected_param, self.to_label = selected_param(self.distribution, param.threshold,param.threshold_label)
        self.threshold = param.threshold
        self.refit = min_LS(matrices, self.selected_param,  intercept=param.formulation.intercept)
        self.save1 = False
        self.save2 = False
        self.save3 = False
        self.method = param.method
        self.formulation = name_formulation
        self.label = label
        self.time = time() - t0

    def __repr__(self):

        string = "\n STABILITY SELECTION : "

        D, Dpath, selected = self.distribution, self.distribution_path, self.selected_param
        unselected = [not i for i in selected]
        Dselected, Dunselected  = np.zeros(len(D)), np.zeros(len(D))
        Dselected[selected], Dunselected[unselected] = D[selected], D[unselected]
        
        plt.bar(range(len(Dselected)), Dselected, color='r', label='selected coefficients')
        plt.bar(range(len(Dunselected)), Dunselected, color='b', label='unselected coefficients')
        plt.axhline(y=self.threshold, color='g',label='Threshold : thresh = '+ str(self.threshold))

        plt.xticks(ticks = np.where(self.to_label)[0], labels = self.label[self.to_label], rotation=30)
        plt.xlabel(StabSel_graph["xlabel"]), plt.ylabel(StabSel_graph["ylabel"]), plt.title(StabSel_graph["title"] + self.method + " using " + self.formulation), plt.legend()

        if (type(self.save1) == str): plt.savefig(self.save1)

        plt.show()
        


        if (type(Dpath) != str):
            lambdas = self.lambdas_path
            N = len(lambdas)
            for i in range(len(selected)):
                if selected[i]: c='r'
                else :          c='b'
                plt.plot(lambdas, [Dpath[j][i] for j in range(N)], c)
            p1 = mpatches.Patch(color='red', label='selected coefficients')
            p2 = mpatches.Patch(color='blue',label='unselected coefficients')
            p3 = mpatches.Patch(color='green',label='Threshold : thresh = '+ str(self.threshold))
            plt.legend(handles=[p1, p2, p3], loc=1)
            plt.axhline(y=self.threshold,color='g')
            plt.xlabel(StabSel_path["xlabel"]), plt.ylabel(StabSel_path["ylabel"]), plt.title(StabSel_path["title"] +  self.method + " using " + self.formulation)
            if (type(self.save2)==str):plt.savefig(self.save2)
            plt.show()

        plt.bar(range(len(self.refit)), self.refit)
        plt.xlabel(StabSel_beta["xlabel"]), plt.ylabel(StabSel_beta["ylabel"]), plt.title(StabSel_beta["title"])
        plt.xticks(np.where(self.selected_param)[0],self.label[self.selected_param], rotation=30)
        if (type(self.save3) == str): plt.savefig(self.save3)
        plt.show()


        string += "\n   Selected variables :  " 
        for i in np.where(selected)[0] :
            string += self.label[i] + "    "

        string += "\n   Running time :  "  + str(round(self.time, 3)) + "s"
        return string

#Here, the main function used is Classo ; from the file compact_func
class solution_LAMfixed:
    ''' Class giving characteristics of the lasso computed
    which also contains a method _repr_() that plot this solution.

    Attributes:
        lambdamax (float) : lambda maximum for which the solution is non-null
        rescaled_lam (bool) : if True, the problem had been computed for lambda*lambdamax (so lambda should be between 0 and 1)
        lambda (float) : lambda for which the problem is solved
        beta (numpy.ndarray) : solution beta of classo
        sigma (float) : solution sigma of classo when formulation is 'R2' or 'R4'
        selected_param (numpy.ndarray) : boolean arrays of size d with True when the variable is selected (which is the case when the i-th component solution of the classo is non-null)
        refit (numpy.ndarray) : solution beta after solving unsparse problem over the set of selected variables.
        formulation (str) : can be 'R1' ; 'R2' ; 'R3' ; 'R4' ; 'C1' ; 'C2'
        time (float) : running time of this action

    '''
    def __init__(self, matrices, param, formulation, numerical_method, label):
        t0 = time()
        self.formulation = formulation
        # Formulation choosing
        if param.formulation == 'not specified': param.formulation = formulation
        if param.numerical_method == "not specified" : param.numerical_method = numerical_method
        name_formulation = param.formulation.name()

        rho = param.formulation.rho
        rho_classification = param.formulation.rho_classification
        e = param.formulation.e
        # Compute the theoretical lam if necessary
        if param.lam == 'theoretical' or param.lam < 0:
            self.lam = param.theoretical_lam
        else:
            self.lam = param.lam

        # Algorithmic method choosing
        numerical_method = choose_numerical_method(param.numerical_method, 'LAM', param.formulation, lam=self.lam)
        param.numerical_method = numerical_method
        self.rescaled_lam = param.rescaled_lam

        # Compute the solution and is the formulation is concomitant, it also compute sigma
        out = Classo(
            matrices, self.lam, typ=name_formulation, meth=numerical_method, rho=rho,
            get_lambdamax=True, true_lam= not self.rescaled_lam, e=e, rho_classification=rho_classification, w=param.formulation.w, intercept = param.formulation.intercept)

        if param.formulation.concomitant: self.lambdamax, self.beta, self.sigma = out
        else: self.lambdamax, self.beta = out
        if param.threshold is None : 
            param.threshold = np.mean(abs(self.beta))

        
        self.selected_param = abs(self.beta) > param.threshold
        self.refit = min_LS(matrices, self.selected_param, intercept=param.formulation.intercept)
        self.label = label
        self.time = time() - t0
        self.save = False

    def __repr__(self):

        string = "\n LAMBDA FIXED : "


        plt.bar(range(len(self.beta)), self.beta), plt.title(LAM_beta["title"] + str(round(self.lam,3) ) ), plt.xlabel(LAM_beta["xlabel"]),plt.ylabel(LAM_beta["ylabel"])
        plt.xticks(np.where(self.selected_param)[0],self.label[self.selected_param], rotation=30)
        if(type(self.save)==str): plt.savefig(self.save)
        plt.show()
        if(self.formulation.concomitant) : 
            string += "\n   Sigma  =  " + str(round(self.sigma, 3))
          
        string += "\n   Selected variables :  " 
        for i in np.where(self.selected_param)[0] :
            string += self.label[i] + "    "

        string += "\n   Running time :  "  + str(round(self.time, 3)) + "s"
        return string







def choose_numerical_method(method, model, formulation, StabSelmethod=None, lam=None):
    ''' Annex function in order to choose the right numerical method, if the given one is invalid

    Args:
        method (str) :
        model (str) :
        formulation (classo_formulation) :
        StabSelmethod (str, optional) :
        lam (float, optional) :

    Returns :
        str : method that should be used.

    '''

    if (formulation.classification): return ('Path-Alg')

    # cases where we use classo at a fixed lambda    
    elif (model == 'LAM') or (model == 'StabSel' and StabSelmethod == 'lam'):

        if formulation.concomitant :
            if not method in ['Path-Alg', 'DR']:
                if (lam > 0.05):
                    return 'Path-Alg'
                else:
                    return 'DR'


        else:
            if not method in ['Path-Alg', 'DR', 'P-PDS', 'PF-PDS']:
                if (lam > 0.1):
                    return 'Path-Alg'
                else:
                    return 'DR'



    # cases where we use pathlasso
    else:
        if formulation.classification:
            if not method in ['Path-Alg', 'DR', 'P-PDS']: return 'Path-Alg'

        elif formulation.concomitant:
            if not method in ['Path-Alg', 'DR']: 
                if formulation.huber : return 'DR'
                else : return 'Path-Alg'

        else:
            if not method in ['Path-Alg', 'DR', 'P-PDS', 'PF-PDS']: return 'Path-Alg'

    return method
















CV_beta             = {
                            "title"  : r"Refitted coefficients after CV model selection" ,
                            "xlabel" : r"Coefficient index $i$" ,
                            "ylabel" : r"Coefficients $\beta_i$ "}
CV_graph            = {
                            "title"  : r" " ,
                            "xlabel" : r"$\lambda / \lambda_{max}$" ,
                            "ylabel" : r"Mean-Squared Error (MSE) "}
LAM_beta            = {
                            "title"  : r"Coefficients at theoretical $\lambda$ = " ,
                            "xlabel" : r"Coefficient index $i$" ,
                            "ylabel" : r"Coefficients $\beta_i$ "}
PATH_beta_path      = {
                            "title"  : r"Coefficients across $\lambda$-path using " ,
                            "xlabel" : r"$\lambda$" ,
                            "ylabel" : r"Coefficients $\beta_i$ "}
PATH_sigma_path     = {
                            "title"  : r"Scale estimate across $\lambda$-path using " ,
                            "xlabel" : r"$\lambda$" ,
                            "ylabel" : r"Scale $\sigma$ "}
StabSel_graph       = {
                            "title"  : r"Stability selection profile of type " ,
                            "xlabel" : r"Coefficient index $i$" ,
                            "ylabel" : r"Selection probability "}
StabSel_path        = {
                            "title"  : r"Stability selection profile across $\lambda$-path using " ,
                            "xlabel" : r"$\lambda$" ,
                            "ylabel" : r"Selection probability "}
StabSel_beta        = {
                            "title"  : r"Refitted coefficients after stability selection" ,
                            "xlabel" : r"Coefficient index $i$" ,
                            "ylabel" : r"Coefficients $\beta_i$ "}

