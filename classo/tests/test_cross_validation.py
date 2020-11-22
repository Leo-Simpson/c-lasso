import numpy as np

from ..cross_validation import (
    train_test_CV,
    train_test_i,
    average_test,
    accuracy_func,
    misclassification_rate,
    CV
)


def test_train_test_CV_non_divisible():
    n = 16
    k = 3

    result = train_test_CV(n,k)


    assert len(result) == k
    assert len(result[0]) == 5 and len(result[1]) == 5 and len(result[2]) == 6
    assert set(np.concatenate(result)) == set(range(n))

def test_train_test_CV_divisible():
    n = 15
    k = 3

    result = train_test_CV(n,k)


    assert len(result) == k
    assert len(result[0]) == 5 and len(result[1]) == 5 and len(result[2]) == 5
    assert set(np.concatenate(result)) == set(range(n))


def test_train_test_i():
    value = [[1,0],[3,4],[2,5]]
    i = 1

    result1, result2 = train_test_i(value, i)

    exp1, exp2 = [0, 1, 2, 5], [3, 4]

    assert len(result1) == len(exp1)
    assert set(exp1) == set(result1)

    assert len(result2) == len(exp2)
    assert set(exp2) == set(result2)

def test_accuracy_func():
    A = np.eye(5)*2
    y = np.array([2, 5, 2, 4, -1])*2 + 20
    beta = np.array([20, 2, 5, 2, 4,-1]) + np.array([0, 1, 1, 1, 1, 1])/2

    result = accuracy_func(
        A, y, beta, typ="R1", intercept=True
    )

    exp = 1

    assert np.isclose(result, exp)

def test_misclassification_rate():
    A = np.eye(6)*2
    y = np.array([1, -1, 1, -1, 1, -1])
    beta = np.ones(6)

    result = misclassification_rate(A, y, beta)

    exp = 0.5

    assert np.isclose(result, exp)

def test_average_test():

    assert True