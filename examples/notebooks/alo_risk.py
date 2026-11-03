# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.2",
#     "c-lasso",
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# [tool.uv.sources]
# c-lasso = { path = "../..", editable = true }
# ///
"""marimo version of `alo.ipynb`.

Run with ``uvx marimo edit --sandbox examples/notebooks/alo_risk.py``.

The Jupyter notebook imports helpers (``generate_data``, ``solve_standard``,
``solve_loo``) that only ever existed as commented-out code in ``classo/alo.py``,
so it no longer runs. This version reproduces the same experiment with the public
API: ``pathlasso`` for the lambda-path and ``classo.alo.alo_classo_risk`` for the
ALO estimate.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.linalg

    from classo import pathlasso
    from classo.alo import alo_classo_risk


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Approximate leave-one-out (ALO) risk

    Along the lambda-path of the constrained Lasso (formulation R1), we compare
    three estimates of the prediction risk:

    - **oos** — the true out-of-sample risk, known here because the data are synthetic;
    - **loo** — leave-one-out cross validation, which refits the path once per sample;
    - **alo** — the approximate leave-one-out risk, computed from the full-data path alone.

    The original notebook used $n = 800$, $p = 400$, $k = 160$, $d = 2$; leave-one-out
    then needs 800 path fits. The defaults below are smaller so the notebook runs in
    seconds — move the sliders to scale up.
    """)
    return


@app.cell
def _():
    n_slider = mo.ui.slider(50, 800, step=50, value=200, label="n (observations)")
    p_slider = mo.ui.slider(20, 400, step=20, value=100, label="p (parameters)")
    mo.vstack([n_slider, p_slider])
    return n_slider, p_slider


@app.function
def generate_data(n, p, k, d, sigma=1.0, seed=None):
    """Random c-lasso problem: k non-zero coefficients, d random constraints."""
    rng = np.random.Generator(np.random.Philox(seed))
    X = rng.normal(scale=1 / np.sqrt(k), size=(n, p))
    C = rng.normal(size=(d, p))
    beta_nz = np.ones(k)
    C_k = C[:, :k]
    # project the non-zero part onto the constraint C beta = 0
    beta_nz = beta_nz - C_k.T @ scipy.linalg.lstsq(C_k.T, beta_nz)[0]
    beta_nz /= np.mean(beta_nz**2)
    beta = np.concatenate((beta_nz, np.zeros(p - k)))
    y = X @ beta + rng.normal(scale=sigma, size=(n,))
    return (X, C, y), beta


@app.function
def risk_outsample(beta, beta_truth, k, sigma):
    """True prediction risk for a new x ~ N(0, I / k)."""
    return np.sum((beta - beta_truth) ** 2, axis=-1) / k + sigma**2


@app.cell
def _(n_slider, p_slider):
    n, p = n_slider.value, p_slider.value
    k, d, sigma = int(0.4 * p), 2, 0.5
    (X, C, y), beta_truth = generate_data(n, p, k, d, sigma, seed=43)

    # lambda-path on the full data, as fractions of lambda_max
    beta_path, lambdas = pathlasso(
        (X, C, y), lambdas=np.logspace(0, 1, num=80, base=1e-3), typ="R1"
    )
    lambdas = np.asarray(lambdas)
    return C, X, beta_path, beta_truth, k, lambdas, n, sigma, y


@app.cell
def _(C, X, lambdas, n, y):
    # leave-one-out: refit the path without sample i, on the same absolute lambdas
    keep = np.ones(n, dtype=bool)
    beta_loo = []
    for i in mo.status.progress_bar(range(n), title="leave-one-out fits"):
        keep[i] = False
        beta_i, _ = pathlasso(
            (X[keep], C, y[keep]), lambdas=lambdas, typ="R1", true_lam=True
        )
        beta_loo.append(beta_i)
        keep[i] = True
    beta_loo = np.stack(beta_loo)  # (n, n_lambdas, p)
    risk_loo = np.mean((y[:, None] - np.einsum("ilp,ip->il", beta_loo, X)) ** 2, axis=0)
    return (risk_loo,)


@app.cell
def _(C, X, beta_path, beta_truth, k, lambdas, risk_loo, sigma, y):
    risk_alo, df = alo_classo_risk(X, C, y, beta_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(lambdas, risk_outsample(beta_path, beta_truth, k, sigma), label="oos")
    ax.loglog(lambdas, risk_loo, label="loo")
    ax.loglog(lambdas, risk_alo, label="alo")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("risk")
    ax.legend()
    fig.suptitle("c-lasso")
    fig
    return


if __name__ == "__main__":
    app.run()
