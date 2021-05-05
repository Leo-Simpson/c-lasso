r"""
BMI prediction using the COMBO dataset 
==========================================

We first consider the `COMBO data set <https://github.com/Leo-Simpson/c-lasso/tree/master/examples/COMBO_data>`_
and show how to predict Body Mass Index (BMI) from microbial genus abundances and two non-compositional covariates  using "filtered_data".
"""

# %%
# Import the package
# ^^^^^^^^^^^^^^^^^^^^
import sys, os
from os.path import join, dirname
classo_dir = dirname(dirname(__file__))
sys.path.append(classo_dir)
from classo import classo_problem, clr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
#  Define how to read csv
# ^^^^^^^^^^^^^^^^^^^^^^^^


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


# %%
#  Load microbiome and covariate data X
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
data_dir = join(classo_dir, "examples/COMBO_data")
X0 = csv_to_np(join(data_dir, "complete_data/GeneraCounts.csv"), begin=0).astype(float)
X_C = csv_to_np(join(data_dir, "CaloriData.csv"), begin=0).astype(float)
X_F = csv_to_np(join(data_dir, "FatData.csv"), begin=0).astype(float)

# %%
#  Load BMI measurements y
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
y = csv_to_np(join(data_dir, "BMI.csv"), begin=0).astype(float)[:, 0]
labels = csv_to_np(join(data_dir, "complete_data/GeneraPhylo.csv")).astype(str)[:, -1]


# %%
# Normalize/transform data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
y = y - np.mean(y)  # BMI data (n = 96)
X_C = X_C - np.mean(X_C, axis=0)  # Covariate data (Calorie)
X_F = X_F - np.mean(X_F, axis=0)  # Covariate data (Fat)
X0 = clr(X0, 1 / 2).T

# %%
# Set up design matrix and zero-sum constraints for 45 genera
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

X = np.concatenate(
    (X0, X_C, X_F), axis=1
)  # Joint microbiome and covariate data and offset
label = np.concatenate([labels, np.array(["Calorie", "Fat"])])
C = np.ones((1, len(X[0])))
C[0, -1], C[0, -2] = 0.0, 0.0


# %%
# Set up c-lassso problem
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

problem = classo_problem(X, y, C, label=label)
problem.formulation.intercept = True
# %%
# Use stability selection with theoretical lambda [Combettes & Müller, 2020b]
problem.model_selection.StabSelparameters.method = "lam"
problem.model_selection.StabSelparameters.threshold_label = 0.5

# %%
# Use formulation R3
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
problem.formulation.concomitant = True

problem.solve()
print(problem)
print(problem.solution)

# %%
# Use formulation R4
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
problem.data.label = label
problem.formulation.huber = True
problem.formulation.concomitant = True

problem.solve()
print(problem)
print(problem.solution)


# %%
# Use formulation R1 with ALO
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ALO is implemented only for R1 without intercept for now.
#

problem.data.label = label
problem.formulation.intercept = False
problem.formulation.huber = False
problem.formulation.concomitant = False
problem.model_selection.ALO = True

problem.solve()
print(problem)
print(problem.solution)
