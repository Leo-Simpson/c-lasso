r"""
Basic example
===============

Let's present what classo does when using its default parameters on synthetic data.

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

m, d, d_nonzero, k, sigma = 100, 200, 5, 1, 0.5
(X, C, y), sol = random_data(m, d, d_nonzero, k, sigma, zerosum=True, seed=1)

# %%
# Remark : one can see the parameters that should be selected :

print(np.nonzero(sol))

# %%
# Define the classo instance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  
# Next we can define a default c-lasso problem instance with the generated data:

problem = classo_problem(X, y, C) 

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
# We only use stability selection as default model selection strategy. 
# The command also allows you to inspect the computed stability profile for all variables 
# at the theoretical λ

problem.solve()

# %%
# Visualisation
# ^^^^^^^^^^^^^^^
#  
# After completion, the results of the optimization and model selection routines 
# can be visualized using

print(problem.solution)