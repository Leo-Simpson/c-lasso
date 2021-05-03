r"""
Advanced example
==================

Let's present how one can specify different aspects of the problem 
formulation and model selection strategy on classo, using synthetic data.

"""

from classo import classo_problem, random_data
import numpy as np

# %%
#  Generate the data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# This code snippet generates a problem instance with sparse ß in dimension
# d=100 (sparsity d_nonzero=5). The design matrix X comprises n=100 samples generated from an i.i.d standard normal
# distribution. The dimension of the constraint matrix C is d x k matrix. The noise level is σ=0.5.
# The input `zerosum=True` implies that C is the all-ones vector and Cß=0. The n-dimensional outcome vector y
# and the regression vector ß is then generated to satisfy the given constraints.
# One can then see the parameters that should be selected.

m, d, d_nonzero, k, sigma = 100, 200, 5, 1, 0.5
(X, C, y), sol = random_data(
    m, d, d_nonzero, k, sigma, zerosum=True, seed=1, intercept=1.0
)
print(np.nonzero(sol))

# %%
# Define the classo instance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Next we can define a default c-lasso problem instance with the generated data:

problem = classo_problem(X, y, C)

# %%
# Change the parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Let's see some example of change in the parameters

problem.formulation.huber = True
problem.formulation.concomitant = False
problem.formulation.intercept = True
problem.model_selection.CV = True
problem.model_selection.LAMfixed = True
problem.model_selection.StabSelparameters.method = "max"
problem.model_selection.CVparameters.seed = 1
problem.model_selection.LAMfixedparameters.rescaled_lam = True
problem.model_selection.LAMfixedparameters.lam = 0.1

# %%
# Check parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You can look at the generated problem instance by typing:

print(problem)

# %%
#  Solve optimization problems
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#  We use stability selection as default model selection strategy.
# The command also allows you to inspect the computed stability profile for all variables
# at the theoretical λ.
# Two other model selections are computed here:
# computation of the solution for a fixed lambda;
# a path computation followed by a computation of the Approximation of the Leave-one Out error (ALO);
# a k-fold cross-validation.

problem.solve()

# %%
# Visualisation
# ^^^^^^^^^^^^^^^
#
# After completion, the results of the optimization and model selection routines
# can be visualized using

print(problem.solution)

# %%
# R1 Formulation with R1
# ^^^^^^^^^^^^^^^
#

problem.formulation.huber = False
problem.model_selection.ALO = True
problem.solve()
print(problem)
print(problem.solution)
