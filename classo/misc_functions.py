import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from scipy.special import erfinv

colo = [
    "red",
    "orange",
    "g",
    "b",
    "pink",
    "y",
    "c",
    "m",
    "purple",
    "yellowgreen",
    "silver",
    "coral",
    "plum",
    "lime",
    "hotpink",
    "palegreen",
    "tan",
    "firebrick",
    "darksalmon",
    "sienna",
    "sandybrown",
    "olive",
    "cadetblue",
    "lawngreen",
    "palevioletred",
    "papayawhip",
    "turquoise",
    "teal",
    "khaki",
    "peru",
    "indianred",
    "brown",
    "slategrey",
]


"""functions required in solver :
rescale, theoretical_lam, min_LS, affichage, check_size, tree_to_matrix
"""


def rescale(matrices):
    """Function that rescale the matrix and returns its scale

    Substract the mean of y, then divides by its norm. Also divide each colomn of X by its norm.
    This will change the solution, not only by scaling it, because then the L1 norm will affect every component equally (and not only the variables with big size)

    Args:
        matrices (tuple) : tuple of three ndarray matrices corresponding to (X,C,y)

    Returns:
        tuple : tuple of the three corresponding matrices after normalization
        tuple : tuple of the three information one need to recover the initial data : lX (list of initial colomn-norms of X), ly (initial norm of y), my (initial mean of y)

    """
    (X, C, y) = matrices

    my = sum(y) / len(y)
    lX = [LA.norm(X[:, j]) for j in range(len(X[0]))]
    ly = LA.norm(y - my * np.ones(len(y)))
    Xn = np.array(
        [[X[i, j] / (lX[j]) for j in range(len(X[0]))] for i in range(len(X))]
    )
    yn = np.array([(y[j] - my) / ly for j in range(len(y))])
    Cn = np.array(
        [[C[i, j] * ly / lX[j] for j in range(len(X[0]))] for i in range(len(C))]
    )
    return ((Xn, Cn, yn), (lX, ly, my))


def theoretical_lam(n, d):
    """Theoretical lambda as a function of the dimensions of the problem

    This function returns (with :math:`\phi = erf`) :

    :math:`4/ \sqrt{n}  \phi^{-1}(1 - 2x)` such that  :math:`x = 4/d ( \phi^{-1}(1-2x)4 + \phi^{-1}(1-2x)^2 )`

    Which is the same (thanks to formula : :math:`norm^{-1}(1-t) = \sqrt{2}\phi^{-1}(1-2t)` ) as :

    :math:`\sqrt{2/n} * norm^{-1}(1-k/p)` such that  :math:`k = norm^{-1}(1 - k/p)^4 + 2norm^{-1}(1 - k/p)^2`

    Args:
        n (int) : number of sample
        d (int)  : number of variables

    Returns:
        float : theoretical lambda

    """
    x = 0.0
    dx = 0.1
    for i in range(10):
        bo = True
        while bo:
            x += dx
            f = erfinv(1 - 2 * x)
            xd = 4 / d * (f ** 4 + f ** 2)
            bo = xd > x
        x = x - dx
        dx = dx / 10
    return 2 * f / np.sqrt(n)


def min_LS(matrices, selected, intercept=False):
    # function to do LS : return  X (X^t X)^-1  X^t y
    X, C, y = matrices
    beta = np.zeros(len(selected))

    if intercept:
        beta[selected] = unpenalized(
            (X[:, selected[1:]], C[:, selected[1:]], y), intercept=selected[0]
        )
    else:
        beta[selected] = unpenalized(
            (X[:, selected], C[:, selected], y), intercept=False
        )

    return beta


def affichage(
    LISTE_BETA,
    path,
    title=" ",
    labels=False,
    pix=False,
    xlabel=" ",
    ylabel=" ",
    naffichage=10,
):
    BETAS = np.array(LISTE_BETA)
    l_index = influence(BETAS, naffichage)
    plt.figure(figsize=(10, 3), dpi=80)
    if pix == "path":
        plt.plot(path, [0] * len(path), "r+")
    plot_betai(labels, l_index, path, BETAS)
    plt.title(title), plt.legend(loc=4, borderaxespad=0.0)
    plt.xlabel(xlabel), plt.ylabel(ylabel)
    if type(pix) == bool and pix:
        plt.matshow(
            [
                [(abs(LISTE_BETA[i][j]) > 1e-2) for i in range(len(LISTE_BETA))]
                for j in range(len(LISTE_BETA[0]))
            ]
        ), plt.show()


def check_size(X, y, C):
    samples, d_in_x = min(len(y), len(X)), len(X[0])
    X2, y2 = X[:samples], y[:samples]
    # if len(y)   >samples   : print("More outputs than features ! ")
    # elif len(X) > samples  : print("More features than outputs !")

    if C is None:
        C2 = np.ones((1, d_in_x))
    else:
        k = len(C)
        d_in_c = len(C[0])
        if d_in_c == d_in_x:
            C2 = C
        elif d_in_c > d_in_x:
            C2 = C[:, :d_in_x]
        else:
            C2 = np.zeros((k, d_in_x ))
            C2[:, : d_in_c] = C

    return X2, y2, C2


def tree_to_matrix(tree, label, with_repr=False):
    # to do here : given a skbio tree and the beta-labels in a given order, return the matrix A and the new labels corresponding
    dicti = dict()
    d = len(label)
    LEAVES = [tip.name for tip in tree.tips()]
    order = (
        []
    )  # list that will give the order in which the codes are added in the dicti, such that it will be easy to remove similar nodes
    for i in range(d):
        name_leaf = label[i]
        dicti[name_leaf] = np.zeros(d)
        dicti[name_leaf][i] = 1
        order.append(name_leaf)
        if name_leaf in LEAVES:
            for n in tree.find(name_leaf).ancestors():
                ancest = n.name
                if ancest[-1] != "_":
                    if ancest not in dicti:
                        dicti[ancest] = np.zeros(d)
                        order.append(ancest)
                    dicti[ancest][i] = 1

    L, label2, tree_repr = [], [], []

    if with_repr:
        for n, l in tree.to_taxonomy():
            nam = n.name
            if nam in label:
                tree_repr.append((nam, l))

    for node in tree.levelorder():
        nam = node.name
        if nam in dicti and nam not in label2:
            label2.append(nam)
            L.append(dicti[nam])

    to_keep = remove_same_vect(L, label2, order)

    return np.array(L)[to_keep].T, np.array(label2)[to_keep], tree_repr


"""functions required in init() :
random_data, csv_to_np, mat_to_np, clr, theoretical_lam, to_zarr
"""


def random_data(
    n,
    d,
    d_nonzero,
    k,
    sigma,
    zerosum=False,
    seed=False,
    classification=False,
    exp=False,
    A=None,
    lb_beta=3,
    ub_beta=10,
    intercept=None,
):
    """Generation of random matrices as data such that y = X.sol + sigma. noise

    The data X is generated as a normal matrix.
    The vector sol is generated randomly with a random support of size d_nonzero,
    and componants are projected random intergers between -10  and 10 on the kernel of C restricted to the support.
    The vector y is then generated with X.dot(sol)+ sigma*noise, with noise a normal vector.


    Args:
        n (int): Number of samples, dimension of y.
        d (int): Number of variables, dimension of sol.
        d_nonzero (int): Number of non null componant of sol.
        k (int) : Number of constraints, number of rows of C.
        sigma (float) : size of the noise.
        zerosum (bool, optional) : If True, then C is the all-one matrix with 1 row, independently of k.
        seed (bool or int, optional) : Seed for random values, for an equal seed, the result will be the same. If set to False: pseudo-random vectors
        classification (bool, optional) : if True, then it returns sign(y) instead of y.
        A (numpy.ndarray) : matrix corresponding to a taxa tree, if it is given, then the problem should be y = X.A.g + eps , C.A.g = 0.

    Returns:
        tuple : tuple of three ndarray that corresponds to the data :  (X,C,y).
        ndarray : array corresponding to sol which is the real solution of the problem y = Xbeta + noise s.t. beta sparse and Cbeta = 0.
    """
    if type(seed) == int:
        np.random.seed(seed)
    else:
        np.random.seed()
    X = np.random.randn(n, d)

    if A is None:
        A = np.eye(d)
    d1 = len(A[0])

    sol, list_i = np.zeros(d1), np.random.randint(d1, size=d_nonzero)

    # sol reduc is random int between lb_beta and ub_beta with a random sign.
    sol_reduc = np.random.randint(low=lb_beta, high=ub_beta + 1, size=d_nonzero) * (
        np.random.randint(2, size=d_nonzero) * 2 - 1
    )

    if zerosum:
        C, k = np.ones((1, d)), 1
    else:
        if k == 0:
            sol[list_i] = sol_reduc
            y = X.dot(A.dot(sol)) + np.random.randn(m) * sigma
            return ((X, np.zeros((0, d1)), y), sol)

        while True:
            C = np.random.randint(low=-1, high=1, size=(k, d))
            if LA.matrix_rank(C) == k:
                break

    while True:
        # building a sparse solution such that C.A sol = 0
        C_reduc = C.dot(A)[:, list_i]
        if LA.matrix_rank(C_reduc) < k:
            list_i = np.random.randint(d1, size=d_nonzero)
            continue
        proj = proj_c(C_reduc, d_nonzero).dot(sol_reduc)
        sol[list_i] = proj
        break

    y = X.dot(A.dot(sol)) + np.random.randn(n) * sigma
    if intercept is not None:
        y = y + intercept
    if classification:
        y = np.sign(y)
    if exp:
        return (np.exp(X), C, y), sol
    return (X, C, y), sol


def csv_to_np(file, begin=1, header=None):
    """Function to read a csv file and to create an ndarray with this

    Args:
        file (str): Name of csv file
        begin (int, optional): First colomn where it should read the matrix
        header (None or int, optional): Same parameter as in the function :func:`pandas.read_csv`

    Returns:
        ndarray : matrix of the csv file
    """
    tab1 = pd.read_csv(file, header=header)
    return np.array(tab1)[:, begin:]


def mat_to_np(file):
    """Function to read a mat file and to create an ndarray with this

    Args:
        file (str): Name of mat file

    Returns:
         ndarray : matrix of the mat file
    """
    arrays = {}
    f = h5py.File(file)
    for k, v in f.items():
        arrays[k] = np.array(v)
    return arrays


def clr(array, coef=0.5):
    """Centered-Log-Ratio transformation

    Set all non positive entry to a constant coef. Then compute the log of each component. Then substract the mean of each column.

    Args:
        array (ndarray) : matrix nxd
        coef (float, optional)  : Value to replace the zero values

    Returns:
        ndarray : clr transformed matrix nxd

    """
    M = np.copy(array)
    null_set = M <= 0.0
    M[null_set] = np.ones(M[null_set].shape) * coef
    M = np.log(M)
    return M - np.mean(M, axis=0)


def to_zarr(obj, name, root, first=True):

    if type(obj) == dict:
        if first:
            zz = root
        else:
            zz = root.create_group(name)

        for key, value in obj.items():
            to_zarr(value, key, zz, first=False)

    elif type(obj) in [np.ndarray, pd.DataFrame]:
        root.create_dataset(name, data=obj, shape=obj.shape)

    elif type(obj) == np.float64:
        root.attrs[name] = float(obj)

    elif type(obj) == np.int64:
        root.attrs[name] = int(obj)

    elif type(obj) == list:
        if name == "tree":
            root.attrs[name] = obj
        else:
            to_zarr(np.array(obj), name, root, first=False)

    elif obj is None or type(obj) in [str, bool, float, int]:
        root.attrs[name] = obj

    else:
        to_zarr(obj.__dict__, name, root, first=first)


"""
misc of compact func
"""


def unpenalized(matrix, eps=1e-3, intercept=False):

    if intercept:
        A1, C1, y = matrix
        A = np.concatenate([np.ones((len(A1), 1)), A1], axis=1)
        C = np.concatenate([np.zeros((len(C1), 1)), C1], axis=1)

    else:
        A, C, y = matrix

    M1 = np.concatenate([A.T.dot(A), C.T], axis=1)
    M2 = np.concatenate([C, np.zeros((len(C), len(C)))], axis=1)
    M = np.concatenate([M1, M2], axis=0)
    b = np.concatenate([A.T.dot(y), np.zeros(len(C))])

    try:
        return LA.inv(M).dot(b)[: len(A[0])]
    except LA.LinAlgError:
        return LA.inv(M + eps * np.eye(len(M))).dot(b)[: len(A[0])]


"""
misc of misc functions
"""


def plot_betai(labels, l_index, path, BETAS):
    j = 0
    for i in range(len(BETAS[0])):
        if j < len(l_index) and i == l_index[j]:
            if not (type(labels) == bool):
                leg = "Coefficient " + str(labels[i])
            else:
                leg = "Coefficient " + str(i)
            plt.plot(path, BETAS[:, i], label=leg, color=colo[j % len(colo)])
            j += 1
        else:
            plt.plot(path, BETAS[:, i], color=colo[(i + j) % len(colo)])


def influence(BETAS, ntop):
    means = np.mean(abs(BETAS), axis=0)
    top = np.argpartition(means, -ntop)[-ntop:]
    return np.sort(top)


def normalize(lb, lna, ly):
    for j in range(len(lb[0])):
        lb[:, j] = lb[:, j] * ly / lna[j]
    return lb


def denorm(B, lna, ly):
    return np.array([ly * B[j] / (np.sqrt(len(B)) * lna[j]) for j in range(len(B))])


def hub(r, rho):
    h = 0
    for j in range(len(r)):
        if abs(r[j]) < rho:
            h += r[j] ** 2
        elif r[j] > 0:
            h += (2 * r[j] - rho) * rho
        else:
            h += (-2 * r[j] - rho) * rho
    return h


def L_LS(A, y, lamb, x):
    return LA.norm(A.dot(x) - y) ** 2 + lamb * LA.norm(x, 1)


def L_conc(A, y, lamb, x):
    return LA.norm(A.dot(x) - y) + np.sqrt(2) * lamb * LA.norm(y, 1)


def L_H(A, y, lamb, x, rho):
    return hub(A.dot(x) - y, rho) + lamb * LA.norm(x, 1)


def proj_c(M, d):
    # Compute I - C^t (C.C^t)^-1 . C : the projection on Ker(C)
    if LA.matrix_rank(M) == 0:
        return np.eye(d)
    return np.eye(d) - LA.multi_dot([M.T, np.linalg.inv(M.dot(M.T)), M])


def remove_same_vect(L, label, order):
    K = len(L)
    to_keep = np.array([True] * K)
    j = label.index(order[0])
    col = L[j]
    for i in range(K - 1):
        new_j = label.index(order[i + 1])
        new_col = L[new_j]
        if np.array_equal(col, new_col):
            to_keep[new_j] = False
        else:
            j, col = new_j, new_col

    return to_keep
