import numpy as np

from ..stability_selection import (
    stability,
    biggest_indexes,
    selected_param,
    build_submatrix,
    build_subset
)

def test_biggest_indexes_empty():
    value = np.array([], dtype=float)
    exp = []

    result = biggest_indexes(value, 0)

    assert set(exp) == set(result)
    assert len(exp) == len(result)

def test_biggest_indexes_empty_big_q():
    value = np.array([], dtype=float)
    exp = []

    result = biggest_indexes(value, 4)

    assert set(exp) == set(result)
    assert len(exp) == len(result)

def test_biggest_indexes_len_less_than_q():
    value = np.array([4, -1, 2], dtype=float)
    exp = [0, 1, 2]

    result = biggest_indexes(value, 4)

    assert set(exp) == set(result)
    assert len(exp) == len(result)

def test_biggest_indexes_len_more_than_q():
    value = np.array([4, -1, 2, 0, -6, 10], dtype=float)
    exp = [5, 4, 0]

    result = biggest_indexes(value, 3)

    assert set(exp) == set(result)
    assert len(exp) == len(result)

def test_biggest_indexes_negative_q():
    value = np.array([1, 2, 3], dtype=float)
    exp = []

    result = biggest_indexes(value, -5)

    assert set(exp) == set(result)
    assert len(exp) == len(result)

def test_build_subset():
    value_n = 20
    value_nS = 5

    result = build_subset(value_n, value_nS)

    assert len(result) == value_nS
    for i in result:
        assert i in range(value_n)

def test_build_submatrix():
    matrix = (np.array([[1,3,2],[5,6,-2]]), np.array([3,6,0]), np.array([1,2]))
    subset = np.array([1])

    result = build_submatrix(matrix, subset)

    exp = (np.array([[5,6,-2]]), np.array([3,6,0]), np.array([2]))

    assert np.all(result[0] == exp[0] )
    assert np.all(result[1] == exp[1] )
    assert np.all(result[2] == exp[2] )

def test_selected_param():
    distribution = np.array([2., 0.1, 7., 14])
    threshold = 10.
    threshold_label = 0.3

    result1, result2 = selected_param(distribution, threshold, threshold_label)

    exp1 = np.array([False, False, False, True])
    exp2 = np.array([True, False, True, True])

    assert np.all(result1 == exp1)
    assert np.all(result2 == exp2)

def test_stability_lam_R1_parameters_independance_and_seed_dependance():

    A = np.ones((10,30))+np.arange(-15,15)+np.arange(-5,5)[:, np.newaxis]
    C = np.zeros((2,30))
    y =  np.arange(10)
    matrix = (A, C, y)

    result1 = stability( matrix,
                        StabSelmethod = "lam",
                        q = 3,
                        B = 20,
                        lam = 0.01,
                        percent_nS = 0.2,
                        formulation = "R1",
                        seed = 14,
                        rho = 6.7,
                        rho_classification = -26.0,
                        true_lam = False,
                        e = 24.0)

    result2 = stability( matrix,
                        StabSelmethod = "lam",
                        q = 3,
                        B = 20,
                        lam = 0.01,
                        percent_nS = 0.2,
                        formulation = "R1",
                        seed = 14,
                        rho = 1.2345,
                        rho_classification = -3.0,
                        true_lam = False,
                        e = 3.0)


    print(result1)
    print(result2)
    assert np.all(result1 == result2)

def test_stability_max_R2_between_0_and_1():

    A = np.ones((10,30))+np.arange(-15,15)+np.arange(-5,5)[:, np.newaxis]
    C = np.zeros((2,30))
    y =  np.arange(10)
    matrix = (A, C, y)

    result = stability( matrix,
                        StabSelmethod = "max",
                        numerical_method = "DR",
                        q = 3,
                        B = 20,
                        percent_nS = 0.2,
                        formulation = "R2",
                        seed = 24,
                        rho = 1.5,
                        rho_classification = -26.0,
                        true_lam = True,
                        e = 24.0)


    assert np.all(result <= 1.)
    assert np.all(result >= 0.)

def test_stability_first_C1_not_too_high_distribution():

    A = np.ones((10,30))+np.arange(-15,15)+np.arange(-5,5)[:, np.newaxis]
    C = np.zeros((2,30))
    y =  np.arange(10)
    matrix = (A, C, y)
    q = 5

    result, _, _ = stability( matrix,
                        StabSelmethod = "first",
                        numerical_method = "P-PDS",
                        q = q,
                        B = 10,
                        percent_nS = 0.2,
                        formulation = "C1",
                        seed = 24,
                        rho = 6.7,
                        rho_classification = -26.0,
                        true_lam = True,
                        e = 24.0)


    assert np.sum(result) <= q