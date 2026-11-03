# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.25.2",
#     "c-lasso",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scipy",
# ]
# [tool.uv.sources]
# c-lasso = { path = "../..", editable = true }
# ///
"""marimo version of `example-notebook.ipynb`.

Run with ``uvx marimo edit --sandbox examples/notebooks/example_notebook.py``.

Differences from the Jupyter notebook:

- The Jupyter notebook mutates one problem per data set from formulation to
  formulation. marimo runs cells by their data dependencies, so each formulation
  here builds its own problem with the settings the notebook had reached by then.
- The Tara section reads ``examples/Tara/tara.npz`` and ``A.npy`` (produced by
  ``examples/Tara/preprocessing.py``) instead of the R data files, so it needs
  neither R nor rpy2.
- Figures are shown in the notebook instead of being saved into ``figures/``.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.io as sio

    from classo import classo_problem, clr


@app.function
def csv_to_np(file, begin=1, header=None):
    """Read a csv file into an ndarray, starting at column ``begin``."""
    return np.array(pd.read_csv(file, header=header))[:, begin:]


@app.function
def show(problem):
    """Solve a problem and display its parameters, solution and figures."""
    problem.solve()
    output = mo.vstack(
        [
            mo.plain_text(repr(problem)),
            mo.plain_text(repr(problem.solution)),
            *problem.solution.plot(),
        ]
    )
    plt.close("all")
    return output


@app.function
def prediction_plot(yhat, y):
    """Scatter plot of predictions against the observed test values."""
    fig, ax = plt.subplots(figsize=(6, 5))
    lo, hi = min(y), max(y)
    ax.plot(yhat, y, "bo", label="sample of the testing set")
    ax.plot([hi, lo], [hi, lo], "k-", label="identity")
    ax.set_xlabel("predictor yhat")
    ax.set_ylabel("real y")
    ax.legend()
    return fig


@app.cell
def _():
    examples_dir = mo.notebook_dir().parent
    return (examples_dir,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # COMBO data

    Combettes, P. L. and Müller, C. L. (2020). Regression models for compositional
    data: General log-contrast formulations, proximal optimization, and microbiome
    data applications. *Statistics in Biosciences*, pages 1–26:

    "The data set comprises the log-transformed relative abundances of p = 87 taxa
    at the genus level across n = 96 healthy subjects. Following previous analyses,
    we investigate the relationship between BMI and the microbial compositions in a
    log-contrast regression framework."

    ## Building matrices
    """)
    return


@app.cell
def _(examples_dir):
    combo_dir = examples_dir / "COMBO_data"
    X0 = csv_to_np(combo_dir / "complete_data/GeneraCounts.csv", begin=0).astype(float)
    X_C = csv_to_np(combo_dir / "CaloriData.csv", begin=0).astype(float)
    X_F = csv_to_np(combo_dir / "FatData.csv", begin=0).astype(float)
    y_combo = csv_to_np(combo_dir / "BMI.csv", begin=0).astype(float)[:, 0]
    labels = csv_to_np(combo_dir / "complete_data/GeneraPhylo.csv").astype(str)[:, -1]

    y_combo = y_combo - np.mean(y_combo)  # BMI data (n = 96)
    X_C = X_C - np.mean(X_C, axis=0)  # covariate data (calories)
    X_F = X_F - np.mean(X_F, axis=0)  # covariate data (fat)
    X0 = clr(X0, 1 / 2).T

    # joint microbiome and covariate data; the covariates are not constrained
    X_combo = np.concatenate((X0, X_C, X_F), axis=1)
    label_combo = np.concatenate([labels, np.array(["Calorie", "Fat"])])
    C_combo = np.ones((1, len(X_combo[0])))
    C_combo[0, -1], C_combo[0, -2] = 0.0, 0.0
    X_combo.shape
    return C_combo, X_combo, combo_dir, label_combo, y_combo


@app.cell
def _(C_combo, X_combo, label_combo, y_combo):
    def combo_problem(concomitant, huber, stabsel_seed=None):
        problem = classo_problem(X_combo, y_combo, C_combo, label=label_combo)
        problem.model_selection.LAMfixed = False
        problem.model_selection.PATH = True
        problem.model_selection.StabSelparameters.method = "lam"
        problem.model_selection.StabSelparameters.threshold_label = 0.5
        if stabsel_seed is not None:
            problem.model_selection.StabSelparameters.seed = stabsel_seed
        problem.formulation.concomitant = concomitant
        problem.formulation.huber = huber
        return problem

    return (combo_problem,)


@app.cell
def _(combo_problem):
    mo.ui.tabs(
        {
            "R1": show(combo_problem(concomitant=False, huber=False)),
            "R2": show(combo_problem(concomitant=False, huber=True)),
            "R3": show(combo_problem(concomitant=True, huber=False)),
            "R4": show(combo_problem(concomitant=True, huber=True, stabsel_seed=123)),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Filtered COMBO data
    """)
    return


@app.cell
def _(combo_dir):
    X_C_f = csv_to_np(combo_dir / "CaloriData.csv", begin=0).astype(float)
    X_F_f = csv_to_np(combo_dir / "FatData.csv", begin=0).astype(float)
    X0_f = csv_to_np(
        combo_dir / "filtered_data/GeneraFilteredCounts.csv", begin=0
    ).astype(float)
    y_f = csv_to_np(combo_dir / "BMI.csv", begin=0).astype(float)[:, 0]
    labels_f = csv_to_np(combo_dir / "filtered_data/GeneraFilteredPhylo.csv").astype(
        str
    )[:, -1]

    y_f = y_f - np.mean(y_f)
    X_C_f = X_C_f - np.mean(X_C_f, axis=0)
    X_F_f = X_F_f - np.mean(X_F_f, axis=0)
    X0_f = clr(X0_f, 1 / 2).T

    X_filtered = np.concatenate((X0_f, X_C_f, X_F_f, np.ones((len(X0_f), 1))), axis=1)
    label_filtered = np.concatenate([labels_f, np.array(["Calorie", "Fat", "Bias"])])
    C_filtered = np.ones((1, len(X_filtered[0])))
    C_filtered[0, -1], C_filtered[0, -2], C_filtered[0, -3] = 0.0, 0.0, 0.0

    def filtered_problem(huber):
        problem = classo_problem(X_filtered, y_f, C_filtered, label=label_filtered)
        problem.model_selection.StabSel = True
        problem.model_selection.StabSelparameters.method = "lam"
        problem.model_selection.StabSelparameters.seed = 2
        problem.formulation.concomitant = True
        problem.formulation.huber = huber
        return problem

    return (filtered_problem,)


@app.cell
def _(filtered_problem):
    mo.ui.tabs(
        {
            "R3": show(filtered_problem(huber=False)),
            "R4": show(filtered_problem(huber=True)),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # pH data

    ## Building matrices
    """)
    return


@app.cell
def _(examples_dir):
    pH = sio.loadmat(examples_dir / "pH_data" / "matlab" / "pHData.mat")
    X_pH, y_uncentered = pH["X"], pH["Y"].T[0]
    y_pH = y_uncentered - np.mean(y_uncentered)  # center y

    def pH_problem(concomitant, huber):
        problem = classo_problem(X_pH, y_pH)
        problem.model_selection.StabSelparameters.method = "lam"
        problem.model_selection.PATH = True
        problem.model_selection.LAMfixed = True
        problem.formulation.concomitant = concomitant
        problem.formulation.huber = huber
        return problem

    X_pH.shape
    return (pH_problem,)


@app.cell
def _(pH_problem):
    mo.ui.tabs(
        {
            "R1": show(pH_problem(concomitant=False, huber=False)),
            "R2": show(pH_problem(concomitant=False, huber=True)),
            "R3": show(pH_problem(concomitant=True, huber=False)),
            "R4": show(pH_problem(concomitant=True, huber=True)),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Tara data

    Bien, J., Yan, X., Simpson, L. and Müller, C. (2020). Tree-Aggregated
    Predictive Modeling of Microbiome Data:

    "Integrative marine data collection efforts such as Tara Oceans (Sunagawa et
    al., 2020) or the Simons CMAP (https://simonscmap.com) provide the means to
    investigate ocean ecosystems on a global scale. Using Tara's environmental and
    microbial survey of ocean surface water (Sunagawa, 2015), we next illustrate how
    trac can be used to globally connect environmental covariates and the ocean
    microbiome. As an example, we learn a global predictive model of ocean salinity
    from n = 136 samples and p = 8916 miTAG OTUs (Logares et al., 2014). trac
    identifies four taxonomic aggregations, the kingdom bacteria and the phylum
    Bacteroidetes being negatively associated and the classes Alpha and
    Gammaproteobacteria being positively associated with marine salinity."

    ## Building matrices

    The training split `tr` is the one used by `2fit_trac_model.R`.
    """)
    return


@app.cell
def _(examples_dir):
    tara_dir = examples_dir / "Tara"
    data = np.load(tara_dir / "tara.npz")
    x_tara, y_tara, tr, label_nodes = data["x"], data["y"], data["tr"], data["label"]
    A = np.load(tara_dir / "A.npy")
    label_short = np.array([label.split("::")[-1] for label in label_nodes])
    te = np.array([i for i in range(len(y_tara)) if i not in tr])

    # aggregate along the taxonomy, as trac does
    pseudo_count = 1
    nleaves = np.sum(A, axis=0)
    logGeom = np.log(pseudo_count + x_tara).dot(A) / nleaves

    def tara_problem():
        problem = classo_problem(logGeom[tr], y_tara[tr], label=label_short)
        problem.formulation.w = 1 / nleaves
        problem.formulation.intercept = True
        problem.formulation.concomitant = False
        return problem

    mo.md(
        f"y: `{y_tara.shape}`, x: `{x_tara.shape}`, A: `{A.shape}`, "
        f"training samples: `{len(tr)}`"
    )
    return label_nodes, logGeom, tara_problem, te, y_tara


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cross validation and path computation
    """)
    return


@app.cell
def _(label_nodes, logGeom, tara_problem, te, y_tara):
    problem_cv = tara_problem()
    problem_cv.model_selection.StabSel = False
    problem_cv.model_selection.PATH = True
    problem_cv.model_selection.CV = True
    problem_cv.model_selection.CVparameters.seed = 6

    cv_output = show(problem_cv)
    cv = problem_cv.solution.CV
    cv_alpha = cv.refit
    mo.vstack(
        [
            cv_output,
            mo.md(
                f"lambda_1SE = `{cv.lambda_1SE:.4f}`, lambda_min = `{cv.lambda_min:.4f}`"
            ),
            mo.md(f"Selected: `{label_nodes[cv.selected_param[1:]].tolist()}`"),
            mo.md("### Prediction plot"),
            prediction_plot(logGeom[te].dot(cv_alpha[1:]) + cv_alpha[0], y_tara[te]),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Stability selection
    """)
    return


@app.cell
def _(label_nodes, logGeom, tara_problem, te, y_tara):
    problem_ss = tara_problem()
    problem_ss.model_selection.StabSel = True
    problem_ss.model_selection.PATH = False
    problem_ss.model_selection.CV = False

    ss_output = show(problem_ss)
    stabsel = problem_ss.solution.StabSel
    ss_alpha = stabsel.refit
    mo.vstack(
        [
            ss_output,
            mo.md(f"Selected: `{label_nodes[stabsel.selected_param[1:]].tolist()}`"),
            mo.md("### Prediction plot"),
            prediction_plot(logGeom[te].dot(ss_alpha[1:]) + ss_alpha[0], y_tara[te]),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
