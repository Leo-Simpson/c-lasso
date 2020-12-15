r"""
BMI prediction using the COMBO dataset 
==========================================

We first consider the `COMBO data set <https://github.com/Leo-Simpson/c-lasso/tree/master/examples/COMBO_data>`_
and show how to predict Body Mass Index (BMI) from microbial genus abundances and two non-compositional covariates  using "filtered_data".
"""

from classo import csv_to_np, classo_problem, clr
import numpy as np
import matplotlib.pyplot as plt

# %%
#  Load microbiome and covariate data X
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

X0  = csv_to_np('COMBO_data/complete_data/GeneraCounts.csv', begin = 0).astype(float)
X_C = csv_to_np('COMBO_data/CaloriData.csv', begin = 0).astype(float)
X_F = csv_to_np('COMBO_data/FatData.csv', begin = 0).astype(float)

# %%
#  Load BMI measurements y
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
y   = csv_to_np('COMBO_data/BMI.csv', begin = 0).astype(float)[:, 0]
labels = csv_to_np('COMBO_data/complete_data/GeneraPhylo.csv').astype(str)[:, -1]


# %%
# Normalize/transform data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
y   = y - np.mean(y) #BMI data (n = 96)
X_C = X_C - np.mean(X_C, axis = 0)  #Covariate data (Calorie)
X_F = X_F - np.mean(X_F, axis = 0)  #Covariate data (Fat)
X0 = clr(X0, 1 / 2).T

# %%
# Set up design matrix and zero-sum constraints for 45 genera
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

X     = np.concatenate((X0, X_C, X_F, np.ones((len(X0), 1))), axis = 1) # Joint microbiome and covariate data and offset
label = np.concatenate([labels, np.array(['Calorie', 'Fat', 'Bias'])])
C = np.ones((1, len(X[0])))
C[0, -1], C[0, -2], C[0, -3] = 0., 0., 0.




# %%
# Set up c-lassso problem
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

problem = classo_problem(X, y, C, label = label) 

# %%
# Use stability selection with theoretical lambda [Combettes & Müller, 2020b]
problem.model_selection.StabSelparameters.method      = 'lam'
problem.model_selection.StabSelparameters.threshold_label = 0.5

# %%
# Use formulation R3
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
problem.formulation.concomitant = True

problem.solve()
print(problem)
print(problem.solution)
plt.tight_layout()

# %%
# Use formulation R4
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
problem.formulation.huber = True
problem.formulation.concomitant = True

problem.solve()
print(problem)
print(problem.solution)
plt.tight_layout()
