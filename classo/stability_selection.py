import numpy as np
import numpy.random as rd
from .compact_func import Classo, pathlasso


"""
Here is the function that does stability selection. It returns the distribution as an d-array.

There is three different stability selection methods implemented here : 'first' ; 'max' ; 'lam'

    - 'first' will compute the whole path until q parameters pop.
               It will then look at those paremeters, and repeat it for each subset of sample.
               It this case it will also return distr_path wich is an n_lam x d - array . It is usefull to have it is one want to plot it.

   - 'max' will do the same but it will stop at a certain lamin that is set at 1e-2 * lambdamax here,
              then will look at the q parameters for which the  max_lam (|beta_i(lam)|) is the highest.

  - 'lam' will, for each subset of sample, compute the classo solution at a fixed lambda. That it will look at the q highest value of |beta_i(lam)|.

"""


def stability(
    matrix,
    StabSelmethod="first",
    numerical_method="Path-Alg",
    Nlam=100,
    lamin=1e-2,
    lam=0.1,
    q=10,
    B=50,
    percent_nS=0.5,
    formulation="LS",
    seed=1,
    rho=1.345,
    rho_classification=-1.0,
    true_lam=False,
    e=1.0,
    w=None,
    intercept=False,
):

    rd.seed(seed)
    n, d = len(matrix[2]), len(matrix[0][0])
    if intercept:
        d += 1
    nS = int(percent_nS * n)
    distribution = np.zeros(d)

    lambdas = np.linspace(1.0, lamin, Nlam)

    if StabSelmethod == "first":

        distr_path = np.zeros((Nlam, d))
        for i in range(B):
            subset = build_subset(n, nS)
            submatrix = build_submatrix(matrix, subset)
            # compute the path until n_active = q.
            BETA = np.array(
                pathlasso(
                    submatrix,
                    lambdas=lambdas,
                    n_active=q + 1,
                    lamin=0,
                    typ=formulation,
                    meth=numerical_method,
                    rho=rho,
                    rho_classification=rho_classification,
                    e=e * percent_nS,
                    w=w,
                    intercept=intercept,
                )[0]
            )

            distr_path = distr_path + (abs(BETA) >= 1e-5)
            # to do : output, instead of lambdas, the average aciv
            """
                distr_path(lambda)_i = 1/B number of time where i is  (among the q-first & activated before lambda) 
            """
        distribution = distr_path[-1]
        return (distribution * 1.0 / B, distr_path * 1.0 / B, lambdas)

    elif StabSelmethod == "lam":

        for i in range(B):
            subset = build_subset(n, nS)
            submatrix = build_submatrix(matrix, subset)
            regress = Classo(
                submatrix,
                lam,
                typ=formulation,
                meth=numerical_method,
                rho=rho,
                rho_classification=rho_classification,
                e=e * percent_nS,
                true_lam=true_lam,
                w=w,
                intercept=intercept,
            )
            if type(regress) == tuple:
                beta = regress[0]
            else:
                beta = regress
            qbiggest = biggest_indexes(abs(beta), q)
            for i in qbiggest:
                distribution[i] += 1

    elif StabSelmethod == "max":

        for i in range(B):
            subset = build_subset(n, nS)
            submatrix = build_submatrix(matrix, subset)
            # compute the path until n_active = q, and only take the last Beta
            BETA = pathlasso(
                submatrix,
                n_active=0,
                lambdas=lambdas,
                typ=formulation,
                meth=numerical_method,
                rho=rho,
                rho_classification=rho_classification,
                e=e * percent_nS,
                w=w,
                intercept=intercept,
            )[0]
            betamax = np.amax(abs(np.array(BETA)), axis=0)
            qmax = biggest_indexes(betamax, q)
            for i in qmax:
                distribution[i] += 1

    return distribution * 1.0 / B


"""
Auxilaries functions that are used in the main function which is stability

"""


# returns the list of the q highest componants of an array, using the fact that it is probably sparse.
def biggest_indexes(array, q):
    qbiggest = []
    nonnul = non_nul_indices(array)
    reduc_array = array[nonnul]
    for i1 in range(q):
        if not np.any(nonnul):
            break
        reduc_index = np.argmax(reduc_array)
        index = nonnul[reduc_index]
        if reduc_array[reduc_index] == 0.0:
            break
        reduc_array[reduc_index] = 0.0
        qbiggest.append(index)
    return qbiggest


# return the list of indices where the componant of the array is null
def non_nul_indices(array):
    L = []
    for i in range(len(array)):
        if not (array[i] == 0.0):
            L.append(i)
    return L


# for a certain threshold, it returns the features that should be selected
def selected_param(distribution, threshold, threshold_label):
    selected, to_label = [False] * len(distribution), [False] * len(distribution)
    for i in range(len(distribution)):
        if distribution[i] > threshold:
            selected[i] = True
        if distribution[i] > threshold_label:
            to_label[i] = True
    return (np.array(selected), np.array(to_label))


# submatrices associated to this subset
def build_submatrix(matrix, subset):
    (A, C, y) = matrix
    subA, suby = A[subset], y[subset]
    return (subA, C, suby)


# random subset of [1,n] of size nS
def build_subset(n, nS):
    return rd.permutation(n)[:nS]
