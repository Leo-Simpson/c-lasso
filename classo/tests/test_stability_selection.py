import numpy as np

from ..stability_selection import stability, biggest_indexes, non_nul_indices, selected_param, build_submatrix, build_subset

# Test of the function biggest_indexes :
# "returns the list of the q highest components of an array, using the fact that it is probably sparse."
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