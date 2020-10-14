path = '/Users/lsimpson/Desktop/GitHub/Figures/exampleCOMBO/'
from CLasso import *
import numpy as np

X0  = csv_to_mat('COMBO_data/GeneraCounts.csv',begin=0).astype(float)
X_C = csv_to_mat('COMBO_data/CaloriData.csv',begin=0).astype(float)
X_F = csv_to_mat('COMBO_data/FatData.csv',begin=0).astype(float)
y   = csv_to_mat('COMBO_data/BMI.csv',begin=0).astype(float)[:,0]
labels  = csv_to_mat('COMBO_data/GeneraPhylo.csv').astype(str)[:,-1]

y   = y - np.mean(y) #BMI data (n=96)
X_C = X_C - np.mean(X_C, axis=0)  #Covariate data (Calorie)
X_F = X_F - np.mean(X_F, axis=0)  #Covariate data (Fat)
X0 = clr(X0, 1 / 2).T

X      = np.concatenate((X0, X_C, X_F, np.ones((len(X0), 1))), axis=1) # Joint microbiome and covariate data and offset
label = np.concatenate([labels,np.array(['Calorie','Fat','Biais'])])
C = np.ones((1,len(X[0])))
C[0,-1],C[0,-2],C[0,-3] = 0.,0.,0.

problem = classo_problem(X,y,C, label=label)

# Solve the problem for a fixed lambda (by default, it will use the theoritical lambda)
problem.model_selection.LAMfixed                    = True
# Solve the stability selection : (by default, it will use the theoritical lambda)
problem.model_selection.StabSelparameters.method      = 'lam'
# Solve the entire path
problem.model_selection.PATH = True

# for R3
problem.solve()
problem.solution.PATH.save = path+'R3-'
problem.solution.StabSel.save1 = path+'R3-StabSel'
problem.solution.StabSel.save3 = path+'R3-StabSel-beta'
print(problem, problem.solution)


# for R4
problem.formulation.huber = True
problem.solve()

problem.solution.PATH.save = path+'R4-'
problem.solution.StabSel.save1 = path+'R4-StabSel'
problem.solution.StabSel.save3 = path+'R4-StabSel-beta'
print(problem, problem.solution)

# for R2
problem.formulation.concomitant = False
problem.solve()
problem.solution.PATH.save = path+'R2-'
problem.solution.StabSel.save1 = path+'R2-StabSel'
problem.solution.StabSel.save3 = path+'R2-StabSel-beta'
print(problem, problem.solution)

# for R1
problem.formulation.huber = False
problem.solve()
problem.solution.PATH.save = path+'R1-'
problem.solution.StabSel.save1 = path+'R1-StabSel'
problem.solution.StabSel.save3 = path+'R1-StabSel-beta'
print(problem, problem.solution)




