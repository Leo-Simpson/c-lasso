r"""
pH prediction using the Central Park soil dataset 
=========================================================


The next microbiome example considers the [Central Park Soil dataset](./examples/CentralParkSoil) from [Ramirez et al.](https://royalsocietypublishing.org/doi/full/10.1098/rspb.2014.1988). The sample locations are shown in the Figure on the right.)

The task is to predict pH concentration in the soil from microbial abundance data.

This task is also done in `Tree-Aggregated Predictive Modeling of Microbiome Data <https://www.biorxiv.org/content/10.1101/2020.09.01.277632v1>`_.

"""
# %%
# Import the package
# ^^^^^^^^^^^^^^^^^^^^
import sys, os
from os.path import join, dirname, abspath
classo_dir = dirname(dirname(abspath("__file__")))
sys.path.append(classo_dir)

from classo import classo_problem
import matplotlib.pyplot as plt
import numpy as np

# %%
#  Load data
# ^^^^^^^^^^^^^^^^^^^
data_dir = join(classo_dir, "examples/CentralParkSoil")
data = np.load(join(data_dir, "cps.npz"))

x = data["x"]
label = data["label"]
y = data["y"]

A = np.load(join(data_dir, "A.npy"))

# %%
#  Preprocess: taxonomy aggregation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

label_short = np.array([l.split("::")[-1] for l in label])

pseudo_count = 1
X = np.log(pseudo_count + x)
nleaves = np.sum(A, axis=0)
logGeom = X.dot(A) / nleaves

n, d = logGeom.shape

tr = np.random.permutation(n)[: int(0.8 * n)]

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
