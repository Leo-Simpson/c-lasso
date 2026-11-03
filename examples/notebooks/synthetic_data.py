# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.2",
#     "c-lasso",
#     "matplotlib",
#     "numpy",
# ]
# [tool.uv.sources]
# c-lasso = { path = "../..", editable = true }
# ///
"""marimo version of `Synthetic data Notebook.ipynb`.

Run with ``uvx marimo edit --sandbox examples/notebooks/synthetic_data.py``.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from classo import classo_problem, random_data


@app.function
def show(problem):
    """Display a solved problem: its parameters, its solution and its figures."""
    return mo.vstack(
        [
            mo.plain_text(repr(problem)),
            mo.plain_text(repr(problem.solution)),
            *problem.solution.plot(),
        ]
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Synthetic data

    ## Basic example

    The c-lasso package includes the routine `random_data` that allows you to
    generate problem instances using normally distributed data.
    """)
    return


@app.cell
def _():
    (X, C, y), sol = random_data(100, 200, 5, 1, 0.5, zerosum=True, seed=1)
    np.nonzero(sol)[0]
    return X, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This generates a problem instance with sparse $\beta$ in dimension $d = 200$
    (sparsity `d_nonzero = 5`). The design matrix $X$ comprises $n = 100$ samples
    drawn i.i.d. from a standard normal distribution, and the noise level is
    $\sigma = 0.5$. With `zerosum=True`, $C$ is the all-ones vector and
    $C\beta = 0$; the outcome $y$ and the regression vector $\beta$ satisfy the
    given constraints.

    Next we define a default c-lasso problem instance with the generated data and
    solve it. The default formulation is R3 with stability selection.
    """)
    return


@app.cell
def _(X, y):
    problem = classo_problem(X, y)
    problem.solve()
    show(problem)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The summary shows the running time(s) for the c-lasso problem instance and
    the variables selected by stability selection. `problem.solution.plot()`
    returns the corresponding figures; set the `save*` attributes of a solution
    (for example `problem.solution.StabSel.save1`) to also write them to disk.

    ## Advanced example

    Next we specify different aspects of the problem formulation and the model
    selection strategy: formulation R2 (Huber loss, no concomitant scale), with
    cross validation, a fixed lambda and the lambda-path in addition to stability
    selection.
    """)
    return


@app.cell
def _():
    (X_adv, C_adv, y_adv), _sol = random_data(
        100, 200, 5, 0, 0.5, zerosum=True, seed=4
    )
    problem_adv = classo_problem(X_adv, y_adv, C_adv)
    problem_adv.formulation.huber = True
    problem_adv.formulation.concomitant = False
    problem_adv.model_selection.CV = True
    problem_adv.model_selection.LAMfixed = True
    problem_adv.model_selection.PATH = True
    problem_adv.model_selection.StabSelparameters.method = "max"
    problem_adv.model_selection.CVparameters.seed = 1
    problem_adv.model_selection.LAMfixedparameters.rescaled_lam = True
    problem_adv.model_selection.LAMfixedparameters.lam = 0.1
    problem_adv.solve()
    show(problem_adv)
    return


if __name__ == "__main__":
    app.run()
