import numpy as np
import matplotlib.pyplot as plt
from ..solver import classo_problem, choose_numerical_method, Formulation
from ..misc_functions import random_data

# in this files, we do test to verify the there is no problem
# in the structure of the parameters, that is implemented in solver
# we verify on different sets of parameters (3 instances, with 4 computation modes and 6 formulations for each)
# that the solver runs. without any error.



# we create a classo instance for tests.

m, d, d_nonzero, k, sigma = 30, 20, 5, 1, 0.5
(X, C, y), sol = random_data(m, d, d_nonzero, k, sigma, zerosum = True, seed = 42)
y = y + 0.2
print(np.nonzero(sol)[0])
close = True


def test_solve_PATH_StabSel_R1():
    aux_test_solve_PATH_StabSel((False, False, False), close_window=close)

def test_solve_PATH_StabSel_C1():
    aux_test_solve_PATH_StabSel((False, False, True), close_window=close)

def test_solve_PATH_StabSel_R3():
    aux_test_solve_PATH_StabSel((True, False, False), close_window=close)


def test_solve_PATH_CV_R1():
    aux_test_solve_PATH_CV((False, False, False), close_window=close)

def test_solve_PATH_CV_R2():
    aux_test_solve_PATH_CV((False, True, False), close_window=close)

def test_solve_PATH_CV_R3():
    aux_test_solve_PATH_CV((True, False, False), close_window=close)

def test_solve_PATH_CV_C2():
    aux_test_solve_PATH_CV((False, True, True), close_window=close)


def test_solve_PATH_LAMfixed_R1():
    aux_test_solve_PATH_LAMfixed((False, False, False), close_window=close)

def test_solve_PATH_LAMfixed_R3():
    aux_test_solve_PATH_LAMfixed((True, False, False), close_window=close)

def test_solve_PATH_LAMfixed_R4():
    aux_test_solve_PATH_LAMfixed((True, True, False), close_window=close)



def test_solve_CV_StabSel_R1():
    aux_test_solve_CV_StabSel((False, False, False), close_window=close)

def test_solve_CV_StabSel_R3():
    aux_test_solve_CV_StabSel((True, False, False), close_window=close)



def test_solve_CV_LAMfixed_R1():
    aux_test_solve_CV_LAMfixed((False, False, False), close_window=close)

def test_solve_CV_LAMfixed_R4():
    aux_test_solve_CV_LAMfixed((True, True, False), close_window=close)

def test_solve_CV_LAMfixed_C1():
    aux_test_solve_CV_LAMfixed((False, False, True), close_window=close)


def test_solve_StabSel_LAMfixed_R2():
    aux_test_solve_StabSel_LAMfixed((False, False, False), close_window=close)

def test_solve_StabSel_LAMfixed_R2():
    aux_test_solve_StabSel_LAMfixed((False, True, False), close_window=close)

def test_solve_StabSel_LAMfixed_C2():
    aux_test_solve_StabSel_LAMfixed((False, True, True), close_window=close)



def aux_test_solve_PATH_StabSel(mode, close_window=True):
    if mode[2]:
        yy = np.sign(y)
    else:
        yy = y 
    pb = classo_problem(X, yy, C= C, label=['label1', 'label2'])
    pb.formulation.intercept = False
    pb.formulation.w = np.array([1.1]*(d//3)+ [1.]*(d-d//3))
    #pb.formulation.rho = 10.

    pb.formulation.concomitant = mode[0]
    pb.formulation.huber = mode[1]
    pb.formulation.classification = mode[2]

    pb.model_selection.PATH = True
    pb.model_selection.CV = False
    pb.model_selection.StabSel =  True
    pb.model_selection.LAMfixed = False

    param = pb.model_selection.PATHparameters
    param.n_active = 20
    param.Nlam = 50
    param.lamin = 1e-3
    param.logscale = True

    param = pb.model_selection.StabSelparameters
    param.seed = 42
    param.method = 'first'
    param.B = 5
    param.q = 5
    param.percent_nS = 0.5
    param.lamin = 0.01
    param.hd = False
    param.lam = 'theoretical'
    param.rescaled_lam = False
    param.threshold = 0.7 
    param.threshold_label = 0.2

    print(pb)
    pb.solve()

    print(pb, pb.solution)
    
    if close_window:
        plt.close("all")
        
def aux_test_solve_PATH_CV(mode, close_window=True):
    if mode[2]:
        yy = np.sign(y)
    else:
        yy = y 
    pb = classo_problem(X, yy, C= C)
    pb.formulation.intercept = True

    pb.formulation.concomitant = mode[0]
    pb.formulation.huber = mode[1]
    pb.formulation.classification = mode[2]

    pb.model_selection.PATH = True
    pb.model_selection.CV = True
    pb.model_selection.StabSel =  False
    pb.model_selection.LAMfixed = False

    param = pb.model_selection.PATHparameters
    param.n_active = 0
    param.lambdas = np.linspace(10., 1e-1, 10)
    param.rescaled_lam = False

    param = pb.model_selection.CVparameters
    param.seed = None
    param.Nsubset = 5
    param.oneSE = True
    param.Nlam = 50
    param.lamin = 1e-3
    param.logscale = True
    print(pb)
    pb.solve()

    print(pb, pb.solution)

    if close_window:
        plt.close("all")

def aux_test_solve_PATH_LAMfixed(mode, close_window=True):
    if mode[2]:
        yy = np.sign(y)
    else:
        yy = y 
    pb = classo_problem(X, yy, C= C)
    pb.formulation.intercept = False

    pb.formulation.concomitant = mode[0]
    pb.formulation.huber = mode[1]
    pb.formulation.classification = mode[2]
    pb.formulation.e = 20
    pb.formulation.rho_scaled = False

    pb.model_selection.PATH = True
    pb.model_selection.CV = False
    pb.model_selection.StabSel =  False
    pb.model_selection.LAMfixed = True

    param = pb.model_selection.PATHparameters
    param.n_active = 0
    param.Nlam = 15
    param.lamin = 0.03
    param.logscale = False

    param = pb.model_selection.LAMfixedparameters
    param.lam = 'theoretical'
    param.rescaled_lam = False
    print(pb)
    pb.solve()

    print(pb, pb.solution)

    if close_window:
        plt.close("all")

def aux_test_solve_CV_StabSel(mode, close_window=True):
    if mode[2]:
        yy = np.sign(y)
    else:
        yy = y 
    pb = classo_problem(X, yy, C= C)
    pb.formulation.intercept = False

    pb.formulation.concomitant = mode[0]
    pb.formulation.huber = mode[1]
    pb.formulation.classification = mode[2]

    pb.model_selection.PATH = False
    pb.model_selection.CV = True
    pb.model_selection.StabSel =  True
    pb.model_selection.LAMfixed = False

    param = pb.model_selection.CVparameters
    param.seed = None
    param.Nsubset = 7
    param.oneSE = False
    param.Nlam = 60
    param.lamin = 1e-2
    param.logscale = False

    param = pb.model_selection.StabSelparameters
    param.seed = None
    param.method = 'max'
    param.B = 70
    param.q = 20
    param.percent_nS = 0.8
    param.lamin = 0.01
    param.hd = False
    param.lam = 'theoretical'
    param.rescaled_lam = False
    param.threshold = 0.5 
    param.threshold_label = 0.2
    print(pb)
    pb.solve()

    print(pb, pb.solution)
    pb.solution.CV.graphic(se_max = 5)

    if close_window:
        plt.close("all")

def aux_test_solve_CV_LAMfixed(mode, close_window=True):
    if mode[2]:
        yy = np.sign(y)
    else:
        yy = y 

    pb = classo_problem(X, yy, C= C)
    pb.formulation.intercept = True
    pb.formulation.w = np.array([1.1]*20+ [1.]*(d-20))

    pb.formulation.concomitant = mode[0]
    pb.formulation.huber = mode[1]
    pb.formulation.classification = mode[2]

    pb.model_selection.PATH = False
    pb.model_selection.CV = True
    pb.model_selection.StabSel =  False
    pb.model_selection.LAMfixed = True

    param = pb.model_selection.CVparameters
    param.seed = 2
    param.Nsubset = 3
    param.oneSE = True
    param.lambdas = np.linspace(1., 1e-1, 20)

    param = pb.model_selection.LAMfixedparameters
    param.lam = 0.1
    param.rescaled_lam = True
    print(pb)
    pb.solve()

    print(pb, pb.solution)

    if close_window:
        plt.close("all")

def aux_test_solve_StabSel_LAMfixed(mode, close_window=True):
    if mode[2]:
        yy = np.sign(y)
    else:
        yy = y 

    pb = classo_problem(X, yy, C= C)
    pb.formulation.intercept = True

    pb.formulation.concomitant = mode[0]
    pb.formulation.huber = mode[1]
    pb.formulation.classification = mode[2]

    pb.model_selection.PATH = False
    pb.model_selection.CV = False
    pb.model_selection.StabSel =  True
    pb.model_selection.LAMfixed = True

    param = pb.model_selection.StabSelparameters
    param.seed = None
    param.method = 'lam'
    param.B = 50
    param.q = 1000
    param.percent_nS = 0.4
    param.lamin = 1.
    param.hd = False
    param.lam = 'theoretical'
    param.rescaled_lam = False
    param.threshold = 0.8
    param.threshold_label = 0.2

    param = pb.model_selection.LAMfixedparameters
    param.lam = 0.
    param.rescaled_lam = False
    print(pb)
    pb.solve()

    print(pb, pb.solution)

    if close_window:
        plt.close("all")




def test_choose_numerical_method_R4DR():
    formulation = Formulation()
    formulation.huber = True
    for model in ['PATH', 'StabSel']:
        value = choose_numerical_method('DR', model, formulation)
        assert value == 'DR'

def test_choose_numerical_method_2():
    formulation = Formulation()
    
    for meth in ["Path-Alg", "DR", "P-PDS", "PF-PDS"]:
        for model in ['PATH', 'StabSel']:
            formulation.concomitant = False
            formulation.classification = False
            formulation.huber = False
            value = choose_numerical_method(meth, model, formulation)
            assert value == meth
            formulation.huber = True
            value = choose_numerical_method(meth, model, formulation)
            assert value == meth
            formulation.classification = True
            value = choose_numerical_method(meth, model, formulation)
            assert value == 'Path-Alg'