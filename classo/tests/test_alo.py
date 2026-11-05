"""Tests for the approximate leave-one-out (ALO) risk and the ALO model selection."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from ..alo import (
    alo_classo_risk,
    alo_cls_h,
    alo_cls_h_naive,
    alo_h,
    solve_cls,
)
from ..misc_functions import random_data
from ..solver import classo_problem

(X, C, y), sol = random_data(40, 8, 3, 1, 0.5, zerosum=True, seed=3)


def null_space_lstsq(X, y, C):
    """Independent constrained least squares: minimise over beta = N z with C N = 0."""
    _, _, vt = np.linalg.svd(C)
    N = vt[np.linalg.matrix_rank(C) :].T
    z, *_ = np.linalg.lstsq(X @ N, y, rcond=None)
    return N @ z


def test_solve_cls_satisfies_constraint_and_matches_null_space_solution():
    beta = solve_cls(X, y, C)
    np.testing.assert_allclose(C @ beta, 0, atol=1e-10)
    np.testing.assert_allclose(beta, null_space_lstsq(X, y, C), atol=1e-8)


def test_alo_cls_h_matches_naive_reference():
    h = alo_cls_h(X, C)
    np.testing.assert_allclose(h, alo_cls_h_naive(X, C), atol=1e-10)
    assert h.shape == (len(y),)
    assert np.all((h >= 0) & (h <= 1))
    # the trace of the constrained hat matrix is p - rank(C)
    assert np.sum(h) == pytest.approx(X.shape[1] - np.linalg.matrix_rank(C))


def test_alo_is_exact_leave_one_out_for_constrained_least_squares():
    # CLS is a linear smoother, so the ALO residuals equal the LOO residuals.
    beta = solve_cls(X, y, C)
    alo_residual, _ = alo_h(X, beta, y, C)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        beta_loo = solve_cls(X[keep], y[keep], C)
        assert alo_residual[i] == pytest.approx(y[i] - X[i] @ beta_loo, rel=1e-8)


def test_alo_h_with_empty_support_returns_plain_residuals():
    residual, h = alo_h(X, np.zeros(X.shape[1]), y, C)
    np.testing.assert_array_equal(residual, y)
    np.testing.assert_array_equal(h, np.zeros(len(y)))


def test_alo_h_restricts_leverage_to_the_support():
    beta = np.zeros(X.shape[1])
    support = np.flatnonzero(sol)
    beta[support] = solve_cls(X[:, support], y, C[:, support])
    _, h = alo_h(X, beta, y, C)
    np.testing.assert_allclose(h, alo_cls_h(X[:, support], C[:, support]))


def test_alo_classo_risk_along_a_path():
    betas = np.array([np.zeros(X.shape[1]), solve_cls(X, y, C)])
    mse, df = alo_classo_risk(X, C, y, betas)
    assert mse.shape == df.shape == (2,)
    assert mse[0] == pytest.approx(np.mean(y**2))
    assert df[0] == 0
    assert df[1] == pytest.approx((X.shape[1] - 1) / len(y))


@pytest.mark.parametrize(
    "option, message",
    [
        ("huber", "huberized"),
        ("classification", "hinge"),
        ("intercept", "unpenalized bias"),
    ],
)
def test_alo_classo_risk_rejects_unsupported_formulations(option, message):
    with pytest.raises(ValueError, match=message):
        alo_classo_risk(X, C, y, np.zeros((1, X.shape[1])), **{option: True})


def alo_problem(concomitant=False):
    pb = classo_problem(X, y, C)
    pb.formulation.concomitant = concomitant
    pb.model_selection.ALO = True
    pb.model_selection.StabSel = False
    param = pb.model_selection.ALOparameters
    param.Nlam = 30
    param.lamin = 1e-2
    return pb


def test_alo_model_selection():
    pb = alo_problem()
    assert "ALO PARAMETERS" in repr(pb)
    assert "with log-scale" in repr(pb.model_selection.ALOparameters)
    pb.solve()
    alo = pb.solution.ALO

    assert alo.BETAS.shape == (30, X.shape[1])
    assert alo.alo.shape == alo.df.shape == (30,)
    assert alo.imin == np.argmin(alo.alo)
    np.testing.assert_array_equal(alo.beta, alo.BETAS[alo.imin])
    np.testing.assert_allclose(C @ alo.beta, 0, atol=1e-10)
    np.testing.assert_array_equal(alo.selected_param, np.abs(alo.beta) > 1e-5)
    assert np.all(alo.refit[~alo.selected_param] == 0)
    assert "ALO COMPUTATION" in repr(alo)
    assert "Selected variables" in repr(alo)


def test_alo_model_selection_with_explicit_lambdas():
    pb = alo_problem()
    param = pb.model_selection.ALOparameters
    param.lambdas = np.linspace(1.0, 0.05, 12)
    param.n_active = 5
    assert "maximum active variables = 5" in repr(param)
    pb.solve()
    assert pb.solution.ALO.alo.shape == (12,)


def test_alo_model_selection_linear_scale():
    pb = alo_problem()
    pb.model_selection.ALOparameters.logscale = False
    assert "with linear-scale" in repr(pb.model_selection.ALOparameters)
    pb.solve()
    np.testing.assert_allclose(
        pb.model_selection.ALOparameters.lambdas, np.linspace(1.0, 1e-2, 30)
    )


def test_alo_plot_returns_and_saves_figures(tmp_path):
    pb = alo_problem()
    pb.solve()
    alo = pb.solution.ALO
    alo.save1 = str(tmp_path / "path-")
    alo.save2 = str(tmp_path / "alo-")
    alo.save3 = str(tmp_path / "refit-")

    figures = alo.plot()
    try:
        # path, ALO profile and refitted coefficients (no sigma path for R1)
        assert len(figures) == 3
        assert all(isinstance(fig, Figure) for fig in figures)
        assert (tmp_path / "path-Beta-path.png").exists()
        assert (tmp_path / "alo-ALO-path.png").exists()
        assert (tmp_path / "refit-ALO-beta.png").exists()
    finally:
        plt.close("all")


def test_alo_model_selection_rejects_concomitant_formulations():
    pb = alo_problem(concomitant=True)
    with pytest.raises(ValueError, match="ALO is not implemented for R3 and R4"):
        pb.solve()
