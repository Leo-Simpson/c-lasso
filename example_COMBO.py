from CLasso import *
import numpy as np



CaloriData           = csv_to_mat('data/CaloriData.csv',begin=0).astype(float)
FatData              = csv_to_mat('data/FatData.csv',begin=0).astype(float)
BMI                  = csv_to_mat('data/BMI.csv',begin=0).astype(float)[:,0]
GeneraFilteredCounts = csv_to_mat('data/GeneraFilteredCounts.csv',begin=0).astype(float)
GeneraCounts         = csv_to_mat('data/GeneraCounts.csv',begin=0).astype(float).T
CFiltered            = sio.loadmat('data/CFiltered.mat')
# load phylogenetic tree
#GeneraPhylo          = mat_to_np('data/GeneraPhylo.mat')['#refs#'].astype(str)

#BMI data (n=96)
y = BMI - np.mean(BMI)
n = len(y)

#Covariate data
X_C = CaloriData - np.mean(CaloriData, axis=0)
X_F = FatData - np.mean(FatData, axis=0)

print(CaloriData[0])

# Predictor labels
#PredLabels = GeneraPhylo[:,7],['Calorie'],['Fat']

# Countdata of 87 genera
# CLR transform data with pseudo count of 0.5 ;
X0 = clr(GeneraCounts, 1 / 2)

# Joint microbiome and covariate data and offset
X = np.concatenate((X0, X_C, X_F, np.ones((n, 1))), axis=1)

# New dimensionality
(n, p) = X.shape
C = np.ones((1,p))
C[0,-1],C[0,-2],C[0,-3] = 0.,0.,0.






problem = classo_problem(X,y,C)


# Solve the problem for a fixed lambda
lam0 = theoritical_lam(n,p)
lam  = lam0 * n
problem.model_selection.LAMfixed = True
problem.model_selection.LAMfixedparameters.lam = lam
problem.model_selection.LAMfixedparameters.true_lam = True


# Solve the stability selection :
pourcent_nS = 0.5
lam0_nS = theoritical_lam(n*pourcent_nS,p)
lam_nS = n*pourcent_nS*lam0_nS
problem.model_selection.SS                       = True
problem.model_selection.SSparameters.pourcent_nS = 0.5
problem.model_selection.SSparameters.method      = 'lam'
problem.model_selection.SSparameters.lam         = lam_nS
problem.model_selection.SSparameters.true_lam    =  True


# Solve the entire path

problem.model_selection.PATH = True

problem.solve()
print(problem)
print(problem.solution)

problem.formulation.huber = True

problem.solve()
print(problem)
print(problem.solution)
