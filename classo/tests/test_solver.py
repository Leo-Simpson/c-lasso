import numpy as np
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ..solver import classo_problem, choose_numerical_method, Formulation
from ..misc_functions import random_data

# In this file, we solve classo_problem instances on different sets of parameters
# (combinations of model selections and formulations) and check that every computed
# solution is consistent: it satisfies the zero-sum constraint C @ beta = 0, has the
# expected shape, and its selected variables agree with its selection rule.


# we create a classo instance for tests.

m, d, d_nonzero, k, sigma = 30, 20, 5, 1, 0.5
(X, C, y), sol = random_data(m, d, d_nonzero, k, sigma, zerosum=True, seed=42)
y = y + 0.2
close = True


def test_solve_PATH_StabSel_R1():
    assert_valid_solution(aux_test_solve_PATH_StabSel((False, False, False), close_window=close))


def test_solve_PATH_StabSel_C1():
    assert_valid_solution(aux_test_solve_PATH_StabSel((False, False, True), close_window=close))


def test_solve_PATH_StabSel_R3():
    assert_valid_solution(aux_test_solve_PATH_StabSel((True, False, False), close_window=close))


def test_solve_PATH_CV_R1():
    assert_valid_solution(aux_test_solve_PATH_CV((False, False, False), close_window=close))


def test_solve_PATH_CV_R2():
    assert_valid_solution(aux_test_solve_PATH_CV((False, True, False), close_window=close))


def test_solve_PATH_CV_R3():
    assert_valid_solution(aux_test_solve_PATH_CV((True, False, False), close_window=close))


def test_solve_PATH_CV_C2():
    assert_valid_solution(aux_test_solve_PATH_CV((False, True, True), close_window=close))


def test_solve_PATH_LAMfixed_R1():
    assert_valid_solution(aux_test_solve_PATH_LAMfixed((False, False, False), close_window=close))


def test_solve_PATH_LAMfixed_R3():
    assert_valid_solution(aux_test_solve_PATH_LAMfixed((True, False, False), close_window=close))


def test_solve_PATH_LAMfixed_R4():
    assert_valid_solution(aux_test_solve_PATH_LAMfixed((True, True, False), close_window=close))


def test_solve_CV_StabSel_R1():
    assert_valid_solution(aux_test_solve_CV_StabSel((False, False, False), close_window=close))


def test_solve_CV_StabSel_R3():
    assert_valid_solution(aux_test_solve_CV_StabSel((True, False, False), close_window=close))


def test_solve_CV_LAMfixed_R1():
    assert_valid_solution(aux_test_solve_CV_LAMfixed((False, False, False), close_window=close))


def test_solve_CV_LAMfixed_R4():
    assert_valid_solution(aux_test_solve_CV_LAMfixed((True, True, False), close_window=close))


def test_solve_CV_LAMfixed_C1():
    assert_valid_solution(aux_test_solve_CV_LAMfixed((False, False, True), close_window=close))


def test_solve_StabSel_LAMfixed_R1():
    assert_valid_solution(aux_test_solve_StabSel_LAMfixed((False, False, False), close_window=close))


def test_solve_StabSel_LAMfixed_R2():
    assert_valid_solution(aux_test_solve_StabSel_LAMfixed((False, True, False), close_window=close))


def test_solve_StabSel_LAMfixed_C2():
    assert_valid_solution(aux_test_solve_StabSel_LAMfixed((False, True, True), close_window=close))


def aux_test_solve_PATH_StabSel(mode, close_window=True):
    if mode[2]:
        yy = np.sign(y)
    else:
        yy = y
    pb = classo_problem(X, yy, C=C, label=["label1", "label2"])
    pb.formulation.intercept = False
    pb.formulation.w = np.array([1.1] * (d // 3) + [1.0] * (d - d // 3))
    # pb.formulation.rho = 10.

    pb.formulation.concomitant = mode[0]
    pb.formulation.huber = mode[1]
    pb.formulation.classification = mode[2]

    pb.model_selection.PATH = True
    pb.model_selection.CV = False
    pb.model_selection.StabSel = True
    pb.model_selection.LAMfixed = False

    param = pb.model_selection.PATHparameters
    param.n_active = 20
    param.Nlam = 50
    param.lamin = 1e-3
    param.logscale = True

    param = pb.model_selection.StabSelparameters
    param.seed = 42
    param.method = "first"
    param.B = 5
    param.q = 5
    param.percent_nS = 0.5
    param.lamin = 0.01
    param.hd = False
    param.lam = "theoretical"
    param.rescaled_lam = False
    param.threshold = 0.7
    param.threshold_label = 0.2

    print(pb)
    pb.solve()

    print(pb, pb.solution)
    assert_plots(pb.solution)

    if close_window:
        plt.close("all")

    return pb


def aux_test_solve_PATH_CV(mode, close_window=True):
    if mode[2]:
        yy = np.sign(y)
    else:
        yy = y
    pb = classo_problem(X, yy, C=C)
    pb.formulation.intercept = True

    pb.formulation.concomitant = mode[0]
    pb.formulation.huber = mode[1]
    pb.formulation.classification = mode[2]

    pb.model_selection.PATH = True
    pb.model_selection.CV = True
    pb.model_selection.StabSel = False
    pb.model_selection.LAMfixed = False

    param = pb.model_selection.PATHparameters
    param.n_active = 0
    param.lambdas = np.linspace(10.0, 1e-1, 10)
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
    assert_plots(pb.solution)

    if close_window:
        plt.close("all")

    return pb


def aux_test_solve_PATH_LAMfixed(mode, close_window=True):
    if mode[2]:
        yy = np.sign(y)
    else:
        yy = y
    pb = classo_problem(X, yy, C=C)
    pb.formulation.intercept = False

    pb.formulation.concomitant = mode[0]
    pb.formulation.huber = mode[1]
    pb.formulation.classification = mode[2]
    pb.formulation.e = 20
    pb.formulation.rho_scaled = False

    pb.model_selection.PATH = True
    pb.model_selection.CV = False
    pb.model_selection.StabSel = False
    pb.model_selection.LAMfixed = True

    param = pb.model_selection.PATHparameters
    param.n_active = 0
    param.Nlam = 15
    param.lamin = 0.03
    param.logscale = False

    param = pb.model_selection.LAMfixedparameters
    param.lam = "theoretical"
    param.rescaled_lam = False
    print(pb)
    pb.solve()

    print(pb, pb.solution)
    assert_plots(pb.solution)

    if close_window:
        plt.close("all")

    return pb


def aux_test_solve_CV_StabSel(mode, close_window=True):
    if mode[2]:
        yy = np.sign(y)
    else:
        yy = y
    pb = classo_problem(X, yy, C=C)
    pb.formulation.intercept = False

    pb.formulation.concomitant = mode[0]
    pb.formulation.huber = mode[1]
    pb.formulation.classification = mode[2]

    pb.model_selection.PATH = False
    pb.model_selection.CV = True
    pb.model_selection.StabSel = True
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
    param.method = "max"
    param.B = 70
    param.q = 20
    param.percent_nS = 0.8
    param.lamin = 0.01
    param.hd = False
    param.lam = "theoretical"
    param.rescaled_lam = False
    param.threshold = 0.5
    param.threshold_label = 0.2
    print(pb)
    pb.solve()

    print(pb, pb.solution)
    assert_plots(pb.solution)
    assert isinstance(pb.solution.CV.graphic(se_max=5), Figure)

    if close_window:
        plt.close("all")

    return pb


def aux_test_solve_CV_LAMfixed(mode, close_window=True):
    if mode[2]:
        yy = np.sign(y)
    else:
        yy = y

    pb = classo_problem(X, yy, C=C)
    pb.formulation.intercept = True
    pb.formulation.w = np.array([1.1] * 20 + [1.0] * (d - 20))

    pb.formulation.concomitant = mode[0]
    pb.formulation.huber = mode[1]
    pb.formulation.classification = mode[2]

    pb.model_selection.PATH = False
    pb.model_selection.CV = True
    pb.model_selection.StabSel = False
    pb.model_selection.LAMfixed = True

    param = pb.model_selection.CVparameters
    param.seed = 2
    param.Nsubset = 3
    param.oneSE = True
    param.lambdas = np.linspace(1.0, 1e-1, 20)

    param = pb.model_selection.LAMfixedparameters
    param.lam = 0.1
    param.rescaled_lam = True
    print(pb)
    pb.solve()

    print(pb, pb.solution)
    assert_plots(pb.solution)

    if close_window:
        plt.close("all")

    return pb


def aux_test_solve_StabSel_LAMfixed(mode, close_window=True):
    if mode[2]:
        yy = np.sign(y)
    else:
        yy = y

    pb = classo_problem(X, yy, C=C)
    pb.formulation.intercept = True

    pb.formulation.concomitant = mode[0]
    pb.formulation.huber = mode[1]
    pb.formulation.classification = mode[2]

    pb.model_selection.PATH = False
    pb.model_selection.CV = False
    pb.model_selection.StabSel = True
    pb.model_selection.LAMfixed = True

    param = pb.model_selection.StabSelparameters
    param.seed = None
    param.method = "lam"
    param.B = 50
    param.q = 1000
    param.percent_nS = 0.4
    param.lamin = 1.0
    param.hd = False
    param.lam = "theoretical"
    param.rescaled_lam = False
    param.threshold = 0.8
    param.threshold_label = 0.2

    param = pb.model_selection.LAMfixedparameters
    param.lam = 0.0
    param.rescaled_lam = False
    print(pb)
    pb.solve()

    print(pb, pb.solution)
    assert_plots(pb.solution)

    if close_window:
        plt.close("all")

    return pb


def test_choose_numerical_method_R4DR():
    formulation = Formulation()
    formulation.huber = True
    for model in ["PATH", "StabSel"]:
        value = choose_numerical_method("DR", model, formulation)
        assert value == "DR"


def test_choose_numerical_method_2():
    formulation = Formulation()

    for meth in ["Path-Alg", "DR", "P-PDS", "PF-PDS"]:
        for model in ["PATH", "StabSel"]:
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
            assert value == "Path-Alg"


def assert_plots(solution):
    """Printing must not draw; plot() must return the figures it draws."""
    before = plt.get_fignums()
    repr(solution)
    assert plt.get_fignums() == before
    figures = solution.plot()
    assert figures
    assert all(isinstance(fig, Figure) for fig in figures)
    assert len(plt.get_fignums()) == len(before) + len(figures)


def test_import_classo_does_not_import_matplotlib():
    code = "import sys, classo; assert 'matplotlib' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


# Exact methods satisfy C @ beta = 0 up to rounding; the iterative ones (DR, P-PDS,
# PF-PDS) only up to their convergence tolerance. Observed worst case when written:
# 1.2e-2 relative, for R4 cross validation with DR.
EXACT_TOL = 1e-10
ITERATIVE_TOL = 5e-2


def assert_constraint(beta, intercept, tol):
    """Check C @ beta = 0 (relative to the l1 norm of beta), ignoring the intercept."""
    beta = np.asarray(beta, dtype=float)
    assert np.all(np.isfinite(beta))
    coefs = beta[1:] if intercept else beta
    assert len(coefs) == d
    assert np.max(np.abs(C @ coefs)) <= tol * max(1.0, np.sum(np.abs(coefs)))


def tolerance(method):
    return EXACT_TOL if method == "Path-Alg" else ITERATIVE_TOL


def assert_refit(solution, intercept):
    """The refit solves least squares on the selected variables only."""
    assert np.all(solution.refit[~solution.selected_param] == 0)
    assert_constraint(solution.refit, intercept, EXACT_TOL)


def assert_valid_solution(pb):
    """Check every model selection that pb.solve() computed."""
    solution, intercept = pb.solution, pb.formulation.intercept
    n_coefs = d + 1 if intercept else d
    computed = 0

    if not isinstance(solution.PATH, str):
        path = solution.PATH
        assert path.BETAS.shape == (len(path.LAMBDAS), n_coefs)
        for beta in path.BETAS:
            assert_constraint(beta, intercept, tolerance(path.method))
        if pb.formulation.concomitant:
            assert np.all(np.isfinite(path.SIGMAS)) and np.all(path.SIGMAS >= 0)
        computed += 1

    if not isinstance(solution.CV, str):
        cv = solution.CV
        assert cv.beta.shape == (n_coefs,)
        method = pb.model_selection.CVparameters.numerical_method
        assert_constraint(cv.beta, intercept, tolerance(method))
        assert np.array_equal(cv.selected_param, np.abs(cv.beta) > 1e-5)
        assert cv.lambda_1SE >= cv.lambda_min
        assert_refit(cv, intercept)
        computed += 1

    if not isinstance(solution.StabSel, str):
        stabsel = solution.StabSel
        assert stabsel.distribution.shape == (n_coefs,)
        assert np.all((stabsel.distribution >= 0) & (stabsel.distribution <= 1))
        assert np.array_equal(
            stabsel.selected_param, stabsel.distribution > stabsel.threshold
        )
        assert_refit(stabsel, intercept)
        computed += 1

    if not isinstance(solution.LAMfixed, str):
        lamfixed = solution.LAMfixed
        param = pb.model_selection.LAMfixedparameters
        assert lamfixed.beta.shape == (n_coefs,)
        assert_constraint(lamfixed.beta, intercept, tolerance(param.numerical_method))
        assert np.array_equal(
            lamfixed.selected_param, np.abs(lamfixed.beta) > param.threshold
        )
        if pb.formulation.concomitant:
            assert np.isfinite(lamfixed.sigma) and lamfixed.sigma >= 0
        assert_refit(lamfixed, intercept)
        computed += 1

    assert computed == 2  # every test configures exactly two model selections
