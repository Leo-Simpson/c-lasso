r"""
Ocean salinity prediction based on marin microbiome data
=========================================================

We repoduce an example of prediction of ocean salinity over ocean microbiome data
that has been introduced in `this article <https://www.biorxiv.org/content/10.1101/2020.09.01.277632v1.full>`_,
where the R package `trac <https://github.com/jacobbien/trac>`_ (which uses c-lasso)
has been used. 



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

# import rpy in order to read data generated on R
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# this code code be used in order to import R library Matrix
utils = importr('utils')    
utils.chooseCRANmirror(ind=1) #
utils.install_packages('Matrix')
importr('Matrix')

#Open R file tara_sal_processed.RDS
file = 'Tara/tara_sal_processed.RDS'
rds = ro.r['readRDS'](file)

rA = ro.r["as.matrix"](rds[4])
x = np.array(rds[1])
y = np.array(rds[0])
A = np.array(rA)

label_OTU = rds[1].colnames
label_sample =rds[1].rownames
label_nodes = np.array(list(rA.colnames))
label_short = np.array([l.split("::")[-1] for l in label_nodes])
print(y.shape)
print(x.shape)
print(A.shape)

# %%
#  Process similar to 2fit_trac_model.R
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# first need to load "tara_sal_trac.Rdata"
# in order to extract training set we are using
ro.r['load']("Tara/tara_sal_trac.RData")


cvfit = ro.r["cvfit"]
cv = cvfit.rx("cv")
lambda_1SE = cv.rx("lambda_1se")

tr = np.array(ro.r['tr']) - 1  # python index starts at 0 when R index starts at 1
te = np.array([i for i in range(len(y)) if not i in tr])


# %%
#  Process similar to trac
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^


pseudo_count = 1
X = np.log(pseudo_count+x)
nleaves = np.sum(A,axis = 0)
logGeom = X.dot(A)/nleaves

# %%
# Cross validation and Path Computation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

problem = classo_problem(logGeom[tr], y[tr], label = label_short)

problem.formulation.w = 1/nleaves
problem.formulation.intercept     = True
problem.formulation.concomitant = False

problem.model_selection.StabSel   = False
problem.model_selection.PATH   = True
problem.model_selection.CV   = True
problem.model_selection.CVparameters.seed = 6 # one could change logscale, Nsubset, oneSE
print(problem)

problem.solve()
print(problem.solution)


selection = problem.solution.CV.selected_param[1:] # exclude the intercept
print(label_nodes[selection])

# %%
# Prediction plot
# """"""""""""""""""""
alpha = problem.solution.CV.refit
yhat = logGeom[te].dot(alpha[1:])+alpha[0]

M1, M2 = max(y[te]), min(y[te])
plt.plot(yhat, y[te], 'bo', label = 'sample of the testing set')
plt.plot([M1, M2], [M1, M2], 'k-', label = "identity")
plt.xlabel('predictor yhat'), plt.ylabel('real y'), plt.legend()
plt.show()

# %%
# Stability selection
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

problem = classo_problem(logGeom[tr], y[tr], label = label_short)

problem.formulation.w = 1/nleaves
problem.formulation.intercept     = True
problem.formulation.concomitant = False


problem.model_selection.PATH   = False
problem.model_selection.CV   = False
# can change q, B, nS, method, threshold etc in problem.model_selection.StabSelparameters

problem.solve()

print(problem, problem.solution)

selection = problem.solution.StabSel.selected_param[1:] # exclude the intercept
print(label_nodes[selection])

# %%
# Prediction plot
# """"""""""""""""""""

alpha = problem.solution.StabSel.refit
yhat = logGeom[te].dot(alpha[1:])+alpha[0]

M1, M2 = max(y[te]), min(y[te])
plt.plot(yhat, y[te], 'bo', label = 'sample of the testing set')
plt.plot([M1, M2],[M1, M2], 'k-', label = "identity")
plt.xlabel('predictor yhat'), plt.ylabel('real y'), plt.legend()
plt.show()