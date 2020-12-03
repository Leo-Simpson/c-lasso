"""
done
"""


import numpy as np

from ..misc_functions import (
    theoretical_lam,
    min_LS,
    affichage,
    influence,
    check_size,
    random_data,
    clr
)


def test_theoretical_lam():
    l1 = theoretical_lam(10,20)
    l2 = theoretical_lam(40,100)
    l3 = theoretical_lam(67,10)

    for l in [l1, l2, l3]:
        assert l <= 1
        assert l > 0

def test_min_LS_no_intercept():
    np.random.seed(123)
    for i in range(5):
        np.random.rand
        A = np.random.randint(10,size=(20, 50))
        C = np.ones((1,50))
        selected = np.random.rand(50)>0.5
        beta = np.random.randint(10,size=50)
        beta = beta - np.mean(beta[selected])
        y = A[:, selected].dot(beta[selected])
        result = min_LS((A,C,y), selected, intercept = False)

        assert np.allclose(A.dot(result),y)

def test_min_LS_intercept():
    np.random.seed(123)
    for i in range(5):
        A = np.random.randint(10,size=(20, 50))
        C = np.ones((1,50))
        selected = np.random.rand(51)>0.5
        beta = np.random.randint(10,size=50)
        beta = beta - np.mean(beta)
        y = A[:, selected[1:]].dot(beta[selected[1:]]) + 10.

        result = min_LS((A,C,y), selected, intercept = True)

        assert np.allclose(A.dot(result[1:])+result[0],y)

def test_affichage():
    nlam = 12
    d = 20
    lBeta = np.random.randint(4, size=(nlam, d) )
    path = np.linspace(1., 0., nlam)
    labels=['a'+str(i) for i in range(d)]
    affichage(lBeta, path, labels=labels)


def test_influence():
    nlam = 13
    betas = np.array([[1, 0, 3 ,4, -3, 0, 0, 0, -6]]*nlam)
    ntop = 4
    exp = np.array([2, 3, 4, 8])

    value = influence(betas, ntop)

    assert np.all( value == exp )

def test_check_size_all_is_good():
    n, d, k = 30, 68, 5
    X = np.ones((n,d))
    y = np.ones(n)
    C = np.ones((k,d))

    Xv, yv, Cv = check_size(X, y, C)

    assert np.all( X == Xv )
    assert np.all( y == yv )
    assert np.all( C == Cv )

def test_check_size_wrong():
    n, d, k = 30, 68, 5
    X = np.ones((n,d))
    y = np.ones(n)
    C = np.zeros((k,d))
    Cwrong = C[:, :d-3]

    Xv, yv, Cv = check_size(X, y, C)

    assert np.all( X == Xv )
    assert np.all( y == yv )
    assert np.all( C == Cv )

def test_random_data_C():
    n = 65
    d = 62
    d_nonzero = 6
    k = 3
    sigma = 0.

    mat, sol = random_data(n, d, d_nonzero, k, sigma, zerosum=False, seed=42)
    X, C, y = mat

    assert np.allclose(C.dot(sol), 0., atol=1e-5)
    assert np.allclose(X.dot(sol), y, atol=1e-5)
    assert len(C) == k

def test_random_data_zerosum():
    n = 65
    d = 62
    d_nonzero = 6
    k = 3
    sigma = 1.

    mat, sol = random_data(n, d, d_nonzero, k, sigma, zerosum=True, seed=42)
    X, C, y = mat

    assert np.isclose(np.mean(sol), 0., atol=1e-4)


def test_clr():
    X = np.eye(10)
    Z = clr(X)
    assert np.allclose(np.mean(Z, axis=0), 0.)