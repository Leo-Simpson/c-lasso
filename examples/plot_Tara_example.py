r"""
Ocean salinity prediction based on marin microbiome data
=========================================================

We repoduce an example of prediction of ocean salinity over ocean microbiome data
that has been introduced in `this article <https://www.biorxiv.org/content/10.1101/2020.09.01.277632v1.full>`_,
where the R package `trac <https://github.com/jacobbien/trac>`_ (which uses c-lasso)
has been used. 

The data come originally from `trac <https://github.com/jacobbien/trac>`_,
then it is preprocessed in python in this `notebook <https://github.com/Leo-Simpson/c-lasso/examples/Tara/preprocess>`_.



Bien, J., Yan, X., Simpson, L. and Müller, C. (2020).
Tree-Aggregated Predictive Modeling of Microbiome Data :

"Integrative marine data collection efforts such as Tara Oceans (Sunagawa et al., 2020)
or the Simons CMAP (https://simonscmap.com)
provide the means to investigate ocean ecosystems on a global scale.
Using Tara’s environmental and microbial survey of ocean surface water (Sunagawa, 2015),
we next illustrate how trac can be used to globally connect environmental covariates
and the ocean microbiome. As an example, we learn a global predictive model of ocean salinity
from n = 136 samples and p = 8916 miTAG OTUs (Logares et al., 2014).
trac identifies four taxonomic aggregations,
the kingdom bacteria and the phylum Bacteroidetes being negatively associated
and the classes Alpha and Gammaproteobacteria being positively associated with marine salinity.
"""

from classo import classo_problem
import matplotlib.pyplot as plt
import numpy as np

# %%
#  Load data
# ^^^^^^^^^^^^^^^^^^^

data = np.load("Tara/tara.npz", allow_pickle=True)

x = data["x"]
label = data["label"]
y = data["y"]
tr = data["tr"]

A = np.load("Tara/A.npy", allow_pickle=True)

# %%
#  Preprocess: taxonomy aggregation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

label_short = np.array([l.split("::")[-1] for l in label])

pseudo_count = 1
X = np.log(pseudo_count + x)
nleaves = np.sum(A, axis=0)
logGeom = X.dot(A) / nleaves


# %%
# Cross validation and Path Computation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

problem = classo_problem(logGeom[tr], y[tr], label=label_short)

problem.formulation.w = 1 / nleaves
problem.formulation.intercept = True
problem.formulation.concomitant = False

problem.model_selection.StabSel = False
problem.model_selection.PATH = True
problem.model_selection.CV = True
problem.model_selection.CVparameters.seed = (
    6  # one could change logscale, Nsubset, oneSE
)
print(problem)

problem.solve()
print(problem.solution)


selection = problem.solution.CV.selected_param[1:]  # exclude the intercept
print(label[selection])

# %%
# Prediction plot
# """"""""""""""""""""

te = np.array([i for i in range(len(y)) if not i in tr])
alpha = problem.solution.CV.refit
yhat = logGeom[te].dot(alpha[1:]) + alpha[0]

M1, M2 = max(y[te]), min(y[te])
plt.plot(yhat, y[te], "bo", label="sample of the testing set")
plt.plot([M1, M2], [M1, M2], "k-", label="identity")
plt.xlabel("predictor yhat"), plt.ylabel("real y"), plt.legend()
plt.tight_layout()

# %%
# Stability selection
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

problem = classo_problem(logGeom[tr], y[tr], label=label_short)

problem.formulation.w = 1 / nleaves
problem.formulation.intercept = True
problem.formulation.concomitant = False


problem.model_selection.PATH = False
problem.model_selection.CV = False
# can change q, B, nS, method, threshold etc in problem.model_selection.StabSelparameters

problem.solve()

print(problem, problem.solution)

selection = problem.solution.StabSel.selected_param[1:]  # exclude the intercept
print(label[selection])

# %%
# Prediction plot
# """"""""""""""""""""

te = np.array([i for i in range(len(y)) if not i in tr])
alpha = problem.solution.StabSel.refit
yhat = logGeom[te].dot(alpha[1:]) + alpha[0]

M1, M2 = max(y[te]), min(y[te])
plt.plot(yhat, y[te], "bo", label="sample of the testing set")
plt.plot([M1, M2], [M1, M2], "k-", label="identity")
plt.xlabel("predictor yhat"), plt.ylabel("real y"), plt.legend()
plt.tight_layout()
