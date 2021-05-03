"""
done
"""


import numpy as np
from numpy.testing import assert_allclose

from ..misc_functions import (
    theoretical_lam,
    min_LS,
    affichage,
    influence,
    check_size,
    random_data,
    clr,
)

from ..path_alg import next_idr2, next_idr1

tol = 1e-5


def test_theoretical_lam():
    l1 = theoretical_lam(10, 20)
    l2 = theoretical_lam(40, 100)
    l3 = theoretical_lam(67, 10)

    for l in [l1, l2, l3]:
        assert l <= 1
        assert l > 0


def test_min_LS_no_intercept():
    np.random.seed(123)
    for i in range(5):
        np.random.rand
        A = np.random.randint(10, size=(20, 50))
        C = np.ones((1, 50))
        selected = np.random.rand(50) > 0.5
        beta = np.random.randint(10, size=50)
        beta = beta - np.mean(beta[selected])
        y = A[:, selected].dot(beta[selected])
        result = min_LS((A, C, y), selected, intercept=False)

        assert_allclose(A.dot(result), y, rtol=tol, atol=tol)


def test_min_LS_intercept():
    np.random.seed(123)
    for i in range(5):
        A = np.random.randint(10, size=(20, 50))
        C = np.ones((1, 50))
        selected = np.random.rand(51) > 0.5
        beta = np.random.randint(10, size=50)
        beta = beta - np.mean(beta)
        y = A[:, selected[1:]].dot(beta[selected[1:]]) + 10.0

        result = min_LS((A, C, y), selected, intercept=True)

        assert_allclose(A.dot(result[1:]) + result[0], y, rtol=tol, atol=tol)


def test_affichage():
    nlam = 12
    d = 20
    lBeta = np.random.randint(4, size=(nlam, d))
    path = np.linspace(1.0, 0.0, nlam)
    labels = ["a" + str(i) for i in range(d)]
    affichage(lBeta, path, labels=labels)


def test_affichage_nolab():
    nlam = 12
    d = 20
    lBeta = np.random.randint(4, size=(nlam, d))
    path = np.linspace(1.0, 0.0, nlam)
    labels = ["a" + str(i) for i in range(d)]
    affichage(lBeta, path, labels=False)


def test_influence():
    nlam = 13
    betas = np.array([[1, 0, 3, 4, -3, 0, 0, 0, -6]] * nlam)
    ntop = 4
    exp = np.array([2, 3, 4, 8])

    value = influence(betas, ntop)

    assert np.all(value == exp)


def test_check_size_all_is_good():
    n, d, k = 30, 68, 5
    X = np.ones((n, d))
    y = np.ones(n)
    C = np.ones((k, d))

    Xv, yv, Cv = check_size(X, y, C)

    assert np.all(X == Xv)
    assert np.all(y == yv)
    assert np.all(C == Cv)


def test_check_size_wrong():
    n, d, k = 30, 68, 5
    X = np.ones((n, d))
    y = np.ones(n)
    C = np.zeros((k, d))

    Cwrong = C[:, : d - 3]

    Xv, yv, Cv = check_size(X, y, Cwrong)

    assert np.all(X == Xv)
    assert np.all(y == yv)
    assert np.all(C == Cv)


def test_check_size_wrong2():
    n, d, k = 45, 49, 2
    Xwrong = np.ones((n + 4, d))
    y = np.ones(n)
    Cwrong = np.zeros((k, d + 3))

    X = Xwrong[:n]
    C = Cwrong[:, :d]

    Xv, yv, Cv = check_size(Xwrong, y, Cwrong)

    assert np.all(X == Xv)
    assert np.all(y == yv)
    assert np.all(C == Cv)


def test_check_size_wrong3():
    n, d = 45, 20
    X = np.ones((n, d))
    ywrong = np.ones(n + 5)
    C = np.ones((1, d))
    y = ywrong[:n]

    Xv, yv, Cv = check_size(X, ywrong, None)

    assert np.all(X == Xv)
    assert np.all(y == yv)
    assert np.all(C == Cv)


def test_random_data_C():
    n = 15
    d = 20
    d_nonzero = 7
    k = 6
    sigma = 0.0
    intercept = 0.2

    for seed in range(10):

        mat, sol = random_data(
            n,
            d,
            d_nonzero,
            k,
            sigma,
            zerosum=False,
            A=np.eye(d),
            seed=seed,
            intercept=intercept,
        )
        X, C, y = mat

        assert_allclose(C.dot(sol), 0.0, rtol=tol, atol=tol)
        assert_allclose(X.dot(sol) + intercept, y, rtol=tol, atol=tol)
        assert len(C) == k


def test_random_data_zerosum():
    n = 65
    d = 62
    d_nonzero = 6
    k = 0
    sigma = 1.0

    mat, sol = random_data(
        n,
        d,
        d_nonzero,
        k,
        sigma,
        zerosum=True,
        exp=True,
        classification=True,
        seed=None,
    )
    X, C, y = mat

    assert set(y).issubset({1, -1})
    assert np.isclose(np.mean(sol), 0.0, atol=tol)


def test_random_data_C_big_k():
    n = 15
    d = 11
    d_nonzero = 10
    k = 10

    mat, sol = random_data(n, d, d_nonzero, k, 0.0, zerosum=False, A=np.eye(d), seed=0)
    X, C, y = mat

    assert_allclose(C.dot(sol), 0.0, rtol=tol, atol=tol)
    assert_allclose(X.dot(sol), y, rtol=tol, atol=tol)
    assert len(C) == k


def test_clr():
    X = np.eye(10)
    Z = clr(X)
    assert_allclose(np.mean(Z, axis=0), 0.0, rtol=tol, atol=tol)


def test_next_idr1():
    mat = np.eye(5)
    l = [True] * 5
    exp = 4
    l[exp] = False
    value = next_idr1(l, mat)
    assert exp == value


def test_next_idr2():
    mat = np.eye(5)
    l = [True] * 5
    exp = 2
    mat[exp, exp] = 0.0
    value = next_idr2(l, mat)
    assert exp == value
