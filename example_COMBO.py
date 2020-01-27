from CLasso import *
import numpy as np


CaloriData           = csv_to_mat('data/CaloriData.csv',begin=0).astype(float)
FatData              = csv_to_mat('data/FatData.csv',begin=0).astype(float)
BMI                  = csv_to_mat('data/BMI.csv',begin=0).astype(float)[:,0]
GeneraFilteredCounts = csv_to_mat('data/GeneraFilteredCounts.csv',begin=0).astype(float)
GeneraCounts         = csv_to_mat('data/GeneraCounts.csv',begin=0).astype(float).T
CFiltered            = sio.loadmat('data/CFiltered.mat')
# load phylogenetic tree
GeneraPhylo          = csv_to_mat('data/GeneraPhylo.csv').astype(str)[:,-1]

labels = np.concatenate([GeneraPhylo,np.array(['Calorie','Fat','Biais'])])
#BMI data (n=96)
y = BMI - np.mean(BMI)
#Covariate data
X_C = CaloriData - np.mean(CaloriData, axis=0)
X_F = FatData - np.mean(FatData, axis=0)

# Countdata of 87 genera
# CLR transform data with pseudo count of 0.5 ;
X0 = clr(GeneraCounts, 1 / 2)
# Joint microbiome and covariate data and offset
X = np.concatenate((X0, X_C, X_F, np.ones((len(X0), 1))), axis=1)
C = np.ones((1,len(X[0])))
C[0,-1],C[0,-2],C[0,-3] = 0.,0.,0.



problem = classo_problem(X,y,C, labels=labels)
problem.formulation.concomitant = True

# Solve the problem for a fixed lambda (by default, it will use the theoritical lambda)
problem.model_selection.LAMfixed = True
problem.model_selection.LAMfixedparameters.true_lam = True


# Solve the stability selection : (by default, it will use the theoritical lambda)
problem.model_selection.SS                       = True
problem.model_selection.SSparameters.method      = 'lam'
problem.model_selection.SSparameters.true_lam    =  True
problem.model_selection.SSparameters.threshold   = 0.6


# Solve the entire path
problem.model_selection.PATH = True
problem.model_selection.PATHparameters.plot_sigma = True


problem.solve()

print(problem)
print(problem.solution)

'''
problem.formulation.huber = True

problem.solve()
print(problem)
print(problem.solution)
'''