import numpy as np

from ..compact_func import (
    pathlasso,
    Classo
)

from ..misc_functions import random_data


m, d, d_nonzero, k, sigma = 30, 20, 5, 1, 0.5
(X, C, y), sol = random_data(m, d, d_nonzero, k, sigma, zerosum = True, seed = 10)

d1 = d//2
w = np.array( [0.9]*d1 + [1.1]*(d-d1))

"""
Test of Classo 
"""

def test_Classo_R2_is_R1_for_high_rho():

    beta1 = Classo((X, C, y), 0.1, typ = "R1", meth="Path-Alg", intercept=True)

    beta2 = Classo((X, C, y), 0.1, typ = "R2", meth="Path-Alg", intercept=True, rho=100)

    assert  np.sum(abs(beta1-beta2))  /  np.sum(abs(beta1)) < 1e-4

def test_Classo_R2_is_R1_for_high_rho_w():

    _, beta1 = Classo((X, C, y), 0.1, typ = "R1", meth="Path-Alg", get_lambdamax=True, w=w)

    beta2 = Classo((X, C, y), 0.1, typ = "R2", meth="Path-Alg", w=w, rho=100)

    assert  np.sum(abs(beta1-beta2))  /  np.sum(abs(beta1)) < 1e-4


def test_Classo_R4_is_R3_for_high_rho():

    beta1 = Classo((X, C, y), 0.1, e=20, typ = "R3", meth="Path-Alg", return_sigm=False)

    beta2 = Classo((X, C, y), 0.1, e=20, typ = "R4", meth="Path-Alg", rho=100, return_sigm=False)
    assert  np.sum(abs(beta1-beta2))  /  np.sum(abs(beta1)) < 1e-4


def test_Classo_C2_is_C1_for_high_rho():
    
    beta1 = Classo((X, C, np.sign(y)), 0.1, typ = "C1")

    beta2 = Classo((X, C, np.sign(y)), 0.1, typ = "C2", rho=-100)

    assert  np.sum(abs(beta1-beta2))  /  np.sum(abs(beta1)) < 1e-4



"""
Test of pathlasso
"""


def test_pathlasso_R1():
    aux_test_pathlasso((X,C,y), "R1", "Path-Alg")

def test_pathlasso_R2():
    aux_test_pathlasso((X,C,y), "R2", "Path-Alg")

def test_pathlasso_R3():
    aux_test_pathlasso((X,C,y), "R3", "Path-Alg")

def test_pathlasso_R4():
    aux_test_pathlasso((X,C,y), "R4", "DR", atol = 1.) # to change : tolerance

def test_pathlasso_C1():
    aux_test_pathlasso((X,C,np.sign(y)), "C1", "Path-Alg")

def test_pathlasso_C2():
    aux_test_pathlasso((X, C, np.sign(y)), "C2", "Path-Alg")





def aux_test_pathlasso(matrix, typ, meth, atol=1e-3):
    betas, lambdas = pathlasso(matrix, typ=typ, meth=meth)

    i = len(lambdas)//2
    lamb  = lambdas[i]
    beta_1 = betas[i]
    beta_2 = Classo(matrix, lamb, typ=typ, true_lam=True, return_sigm=False, meth=meth)
    assert np.allclose(beta_1, beta_2, atol=atol)

