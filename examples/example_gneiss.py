r"""
pH prediction using the 88 soils dataset 
===================================================

The next microbiome example considers a
`Soil dataset <https://github.com/Leo-Simpson/c-lasso/tree/master/examples/pH_data>`_ .

The data are generated thanks to a `qiime2 workflow <https://github.com/Leo-Simpson/c-lasso/blob/master/examples/pH_data/qiime2/read%20data.ipynb>`_
similar to `a gneiss tutorial <https://github.com/biocore/gneiss/blob/master/ipynb/88soils/88soils-qiime2-tutorial.ipynb>`_.

This workflow treat `some files <https://github.com/Leo-Simpson/c-lasso/blob/master/examples/pH_data/qiime2/originals>`_ 
taken from `gneiss GitHub <https://github.com/biocore/gneiss/tree/master/ipynb/88soils>`_.


The task is to predict pH concentration in the soil from microbial abundance data.

A similar analysis is also done in `Tree-Aggregated Predictive Modeling of Microbiome Data <https://www.biorxiv.org/content/10.1101/2020.09.01.277632v1>`_.
 `on another dataset <https://royalsocietypublishing.org/doi/full/10.1098/rspb.2014.1988>`_
"""

import sys, os
from os.path import join, dirname, abspath
classo_dir = dirname(dirname(abspath("__file__")))
sys.path.append(classo_dir)
from classo import classo_problem
import numpy as np
import pandas as pd


# %%
#  Load data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


t = pd.read_csv("pH_data/qiime2/news/table.csv", index_col=0)
metadata = pd.read_table(
    "pH_data/qiime2/originals/88soils_modified_metadata.txt", index_col=0
)
y_uncent = metadata["ph"].values


X = t.values
label = t.columns


# second option to load the data
# import scipy.io as sio
# pH = sio.loadmat("pH_data/matlab/pHData.mat")
# tax = sio.loadmat("pH_data/matlab/taxTablepHData.mat")["None"][0]
# X, y_uncent = pH["X"], pH["Y"].T[0]
# label = None

y = y_uncent - np.mean(y_uncent)  # Center Y
print(X.shape)
print(y.shape)

# %%
# Set up c-lassso problem
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

problem = classo_problem(X, y, label=label)

problem.model_selection.StabSelparameters.method = "lam"
problem.model_selection.PATH = True
problem.model_selection.LAMfixed = True
problem.model_selection.PATHparameters.n_active = X.shape[1] + 1

# %%
# Solve for R1
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
problem.formulation.concomitant = False
problem.solve()
print(problem, problem.solution)

# %%
# Solve for R2
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
problem.formulation.huber = True
problem.solve()
print(problem, problem.solution)


# %%
# Solve for R3
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
problem.formulation.concomitant = True
problem.formulation.huber = False
problem.solve()
print(problem, problem.solution)


# %%
# Solve for R4
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Remark : we reset the numerical method here,
# because it has been automatically set to '¨Path-Alg'
# for previous computations, but for R4, "DR" is much better
# as explained in the documentation, R4 "Path-Alg" is a method for fixed lambda
# but is (paradoxically) bad to compute the lambda-path
# because of the absence of possible warm-start in this method

problem.model_selection.PATHparameters.numerical_method = "DR"
problem.formulation.huber = True
problem.solve()
print(problem, problem.solution)
