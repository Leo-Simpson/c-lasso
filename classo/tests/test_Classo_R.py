import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

from ..compact_func import pathlasso, Classo


from ..solve_R1 import problem_R1, Classo_R1
from ..solve_R2 import problem_R2, Classo_R2
from ..solve_R3 import problem_R3, Classo_R3, pathlasso_R3
from ..solve_R4 import problem_R4, Classo_R4

from ..misc_functions import random_data

tol = 2e-2

m, d, d_nonzero, k, sigma = 30, 20, 5, 1, 0.5
matrices, sol = random_data(m, d, d_nonzero, k, sigma, zerosum=True, seed=42)
X, C, y = matrices

d1 = d // 2
w = np.array([0.9] * d1 + [1.1] * (d - d1))


"""
Test of Classo 
"""


def test_Classo_R1_all_methods_match():

    lam = 0.05
    pb = problem_R1(matrices, "Path-Alg")
    beta_ref = Classo_R1(pb, lam)

    pb = problem_R1(matrices, "DR")
    beta = Classo_R1(pb, lam)
    print(np.sum(abs(beta_ref - beta)) / np.sum(abs(beta_ref)))
    assert_allclose(beta_ref, beta, rtol=tol, atol=tol)
    # assert  np.sum(abs(beta_ref-beta))  /  np.sum(abs(beta_ref)) < tol

    pb = problem_R1(matrices, "P-PDS")
    beta = Classo_R1(pb, lam)
    print(np.sum(abs(beta_ref - beta)) / np.sum(abs(beta_ref)))
    assert_allclose(beta_ref, beta, rtol=tol, atol=tol)
    # assert  np.sum(abs(beta_ref-beta))  /  np.sum(abs(beta_ref)) < tol

    pb = problem_R1(matrices, "PF-PDS")
    beta = Classo_R1(pb, lam)
    print(np.sum(abs(beta_ref - beta)) / np.sum(abs(beta_ref)))
    assert_allclose(beta_ref, beta, rtol=tol, atol=tol)
    # assert  np.sum(abs(beta_ref-beta))  /  np.sum(abs(beta_ref)) < tol


def test_Classo_R1_all_methods_match_no_C():

    matrices2, sol2 = random_data(m, d, d_nonzero, 0, sigma, zerosum=False, seed=41)

    lam = 0.05
    pb = problem_R1(matrices2, "Path-Alg")
    beta_ref = Classo_R1(pb, lam)

    pb = problem_R1(matrices2, "DR")
    beta = Classo_R1(pb, lam)
    print(np.sum(abs(beta_ref - beta)) / np.sum(abs(beta_ref)))
    assert_allclose(beta_ref, beta, rtol=tol, atol=tol)
    # assert  np.sum(abs(beta_ref-beta))  /  np.sum(abs(beta_ref)) < tol

    pb = problem_R1(matrices2, "P-PDS")
    beta = Classo_R1(pb, lam)
    print(np.sum(abs(beta_ref - beta)) / np.sum(abs(beta_ref)))
    assert_allclose(beta_ref, beta, rtol=tol, atol=tol)
    # assert  np.sum(abs(beta_ref-beta))  /  np.sum(abs(beta_ref)) < tol


def test_Classo_R2_all_methods_match():

    rho = 1.345
    lam = 0.1

    pb = problem_R2(matrices, "Path-Alg", rho)
    beta_ref = Classo_R2(pb, lam)

    pb = problem_R2(matrices, "DR", rho)
    beta = Classo_R2(pb, lam)
    print(np.sum(abs(beta_ref - beta)) / np.sum(abs(beta_ref)))
    assert_allclose(beta_ref, beta, rtol=tol, atol=tol)

    pb = problem_R2(matrices, "P-PDS", rho)
    beta = Classo_R2(pb, lam)
    print(np.sum(abs(beta_ref - beta)) / np.sum(abs(beta_ref)))
    assert_allclose(beta_ref, beta, rtol=tol, atol=tol)

    pb = problem_R2(matrices, "PF-PDS", rho)
    beta = Classo_R2(pb, lam)
    print(np.sum(abs(beta_ref - beta)) / np.sum(abs(beta_ref)))
    assert_allclose(beta_ref, beta, rtol=tol, atol=tol)


def test_Classo_R3_all_methods_match():

    lam = 0.1
    pb = problem_R3(matrices, "Path-Alg")
    beta_ref, s_ref = Classo_R3(pb, lam)

    pb = problem_R3(matrices, "DR")
    beta, s = Classo_R3(pb, lam)

    print(np.sum(abs(beta_ref - beta)) / np.sum(abs(beta_ref)))
    assert_allclose(beta_ref, beta, rtol=tol, atol=tol)
    # assert  np.sum(abs(beta_ref-beta))  /  np.sum(abs(beta_ref)) < tol


def test_Classo_R4_all_methods_match():

    lam = 0.1
    rho = 1.345
    pb = problem_R4(matrices, "Path-Alg", rho)
    beta_ref, s_ref = Classo_R4(pb, lam)

    pb = problem_R4(matrices, "DR", rho)
    beta, s = Classo_R4(pb, lam)

    print(np.sum(abs(beta_ref - beta)) / np.sum(abs(beta_ref)))
    assert_allclose(beta_ref, beta, rtol=tol, atol=tol)
    # assert  np.sum(abs(beta_ref-beta))  /  np.sum(abs(beta_ref)) < tol


def test_Classo_lam_null():
    lam = 0.0
    rho = 1.345
    m, d, d_nonzero, k, sigma = 10, 20, 4, 1, 0.5
    matrices, sol = random_data(m, d, d_nonzero, k, sigma, zerosum=True, seed=42)
    X, C, y = matrices

    pb1 = problem_R1(matrices, "Path-Alg")
    beta1 = Classo_R1(pb1, lam)
    assert_allclose(X.dot(beta1), y, rtol=tol, atol=tol)

    pb2 = problem_R2(matrices, "Path-Alg", rho)
    beta2 = Classo_R2(pb2, lam)

    pb3 = problem_R3(matrices, "Path-Alg")
    beta3, _ = Classo_R3(pb3, lam)
    assert_allclose(X.dot(beta3), y, rtol=tol, atol=tol)
