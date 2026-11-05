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
"""marimo version of `Testing parameters.ipynb`.

Run with ``uvx marimo edit --sandbox examples/notebooks/testing_parameters.py``.

The Jupyter notebook mutates three problems in place, cell after cell, and resets
their numerical methods between formulations. marimo runs cells by their data
dependencies rather than top to bottom, so here every formulation builds three
freshly configured problems instead; no reset is needed.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from classo import classo_problem, random_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Testing parameters

    Three problem instances with different numerical methods and model-selection
    parameters are solved for each of the six formulations R1–R4, C1 and C2.

    ## Data
    """)
    return


@app.cell
def _():
    m, d, d_nonzero, k, sigma = 150, 100, 5, 1, 0.5
    (X, C, y), sol = random_data(m, d, d_nonzero, k, sigma, zerosum=True, seed=0)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(len(sol)), sol)
    ax.set_title("True coefficients")
    mo.vstack(
        [
            fig,
            mo.md(f"Non-zero coefficients: `{np.flatnonzero(sol).tolist()}`"),
            mo.md(f"lambda_max = `{max(abs(X.T.dot(y))):.3f}`"),
        ]
    )
    return C, X, d, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Problems

    `make_problems` returns the three configured instances for one formulation:

    | | Path | Cross validation | Stability selection | Lambda fixed |
    |---|---|---|---|---|
    | problem 1 (intercept) | default method, `n_active = 20`, 50 lambdas to 1e-3 | 5 folds, 1SE | `first`, B = 50, q = 10 | P-PDS, theoretical lambda |
    | problem 2 (intercept, weights) | P-PDS, 50 lambdas to 0.1 | 7 folds | `max`, B = 70, q = 20 | DR, lambda = 0.1 |
    | problem 3 | DR, 20 lambdas to 0.1 | P-PDS, 3 folds, 1SE | `lam`, B = 50 | lambda = 0 |
    """)
    return


@app.cell
def _(C, X, d, y):
    def make_problems(concomitant, huber, classification, first_section=False):
        """Three configured problems for one formulation."""
        target = np.sign(y + 0.5) if classification else y + 0.5
        problems = []
        for i in range(3):
            problem = classo_problem(X, target, C=C)
            selection = problem.model_selection
            selection.PATH = selection.CV = selection.StabSel = True
            selection.LAMfixed = True
            problem.formulation.concomitant = concomitant
            problem.formulation.huber = huber
            problem.formulation.classification = classification
            problems.append(problem)

        p1, p2, p3 = problems
        p1.formulation.intercept = True
        p2.formulation.intercept = True
        p3.formulation.intercept = False
        p2.formulation.w = np.array([1.1] * 40 + [1.0] * (d - 40))

        # lambda-path
        for p, method, n_active, lambdas in [
            (p1, " ", 20, np.linspace(1.0, 0.001, 50)),
            (p2, "P-PDS", 0, np.linspace(1.0, 0.1, 50)),
            (p3, "DR", 0, np.linspace(1.0, 0.1, 20)),
        ]:
            param = p.model_selection.PATHparameters
            param.numerical_method = method
            param.n_active = n_active
            param.lambdas = lambdas

        # cross validation
        for p, seed, method, nsubset, one_se, lambdas in [
            (p1, None, " ", 5, True, np.linspace(1.0, 0.001, 80)),
            (p2, None, " ", 7, False, np.linspace(1.0, 0.001, 80)),
            (p3, 2, "P-PDS", 3, True, np.linspace(1.0, 0.1, 20)),
        ]:
            param = p.model_selection.CVparameters
            param.seed = seed
            param.numerical_method = method
            param.Nsubset = nsubset
            param.oneSE = one_se
            param.lambdas = lambdas

        # stability selection
        for p, method, B, q, percent_nS, lamin, threshold in [
            (p1, "first", 50, 10, 0.5, 0.01, 0.7),
            (p2, "max", 70, 20, 0.8, 0.01, 0.5),
            (p3, "lam", 50, 1000, 0.4, 1.0, 0.8),
        ]:
            param = p.model_selection.StabSelparameters
            param.seed = None
            param.numerical_method = " "
            param.method = method
            param.B = B
            param.q = q
            param.percent_nS = percent_nS
            param.lamin = lamin
            param.hd = False
            param.lam = "theoretical"
            param.true_lam = True
            param.threshold = threshold
            param.threshold_label = 0.2

        # lambda fixed (the notebook switches problem 3 to Path-Alg after R1)
        for p, method, lam, true_lam in [
            (p1, "P-PDS", "theoretical", True),
            (p2, "DR", 0.1, False),
            (p3, " " if first_section else "Path-Alg", 0.0, True),
        ]:
            param = p.model_selection.LAMfixedparameters
            param.numerical_method = method
            param.lam = lam
            param.true_lam = true_lam

        return problems

    def solve_and_show(problems):
        """Solve each problem and show parameters, solution and figures per problem."""
        panels = {}
        for i, problem in enumerate(problems, start=1):
            problem.solve()
            panels[f"problem {i}"] = mo.vstack(
                [
                    mo.plain_text(repr(problem)),
                    mo.plain_text(repr(problem.solution)),
                    *problem.solution.plot(),
                ]
            )
            plt.close("all")
        return mo.accordion(panels)

    return make_problems, solve_and_show


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## R1
    """)
    return


@app.cell
def _(make_problems, solve_and_show):
    solve_and_show(make_problems(False, False, False, first_section=True))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## R2
    """)
    return


@app.cell
def _(make_problems, solve_and_show):
    solve_and_show(make_problems(False, True, False))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## R3
    """)
    return


@app.cell
def _(make_problems, solve_and_show):
    solve_and_show(make_problems(True, False, False))
    return


@app.cell(hide_code=True)
def _():
    run_r4 = mo.ui.run_button(label="Run R4 (about one hour)")
    mo.vstack(
        [
            mo.md(r"""
    ## R4

    R4 computes its paths with the iterative DR method. With these parameters,
    problem 2's stability selection alone (70 subsamples, each a full DR path)
    takes about 45 minutes, and the section about an hour in total, so it only
    runs on request.
    """),
            run_r4,
        ]
    )
    return (run_r4,)


@app.cell
def _(make_problems, run_r4, solve_and_show):
    mo.stop(not run_r4.value, mo.md("*Press the button above to run R4.*"))
    solve_and_show(make_problems(True, True, False))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## C1

    Classification uses the signs of the outcome.
    """)
    return


@app.cell
def _(make_problems, solve_and_show):
    solve_and_show(make_problems(False, False, True))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## C2
    """)
    return


@app.cell
def _(make_problems, solve_and_show):
    solve_and_show(make_problems(False, True, True))
    return


if __name__ == "__main__":
    app.run()
