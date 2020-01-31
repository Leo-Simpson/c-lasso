from CLasso import *
import numpy as np

X0  = csv_to_mat('data/GeneraCounts.csv',begin=0).astype(float).T
X_C = csv_to_mat('data/CaloriData.csv',begin=0).astype(float)
X_F = csv_to_mat('data/FatData.csv',begin=0).astype(float)
y   = csv_to_mat('data/BMI.csv',begin=0).astype(float)[:,0]
labels  = csv_to_mat('data/GeneraPhylo.csv').astype(str)[:,-1]

y   = y - np.mean(y) #BMI data (n=96)
X_C = X_C - np.mean(X_C, axis=0)  #Covariate data (Calorie)
X_F = X_F - np.mean(X_F, axis=0)  #Covariate data (Fat)
X0 = clr(X0, 1 / 2)

X      = np.concatenate((X0, X_C, X_F, np.ones((len(X0), 1))), axis=1) # Joint microbiome and covariate data and offset
labels = np.concatenate([labels,np.array(['Calorie','Fat','Biais'])])
C = np.ones((1,len(X[0])))
C[0,-1],C[0,-2],C[0,-3] = 0.,0.,0.

problem = classo_problem(X,y,C, labels=labels)

# Solve the problem for a fixed lambda (by default, it will use the theoritical lambda)
problem.model_selection.LAMfixed                    = True
problem.model_selection.LAMfixedparameters.true_lam = True

# Solve the stability selection : (by default, it will use the theoritical lambda)
problem.model_selection.SS                       = True
problem.model_selection.SSparameters.method      = 'lam'
problem.model_selection.SSparameters.true_lam    =  True
problem.model_selection.SSparameters.threshold   = 0.7
problem.model_selection.SSparameters.threshold_label   = 0.4

# Solve the entire path
problem.model_selection.PATH = True
problem.model_selection.PATHparameters.plot_sigma = True

problem.solve()
print(problem)
print(problem.solution)


problem.formulation.huber = True
# We don't solve the entire path here
problem.model_selection.PATH = False
problem.solve()
print(problem)
print(problem.solution)

''' 
print(np.linalg.norm(y)/np.sqrt(n/2))

BETAS = np.array(problem.solution.PATH.BETAS)
SIGMAS= np.array(problem.solution.PATH.SIGMAS)
print( np.all(  np.isclose(np.linalg.norm(BETAS.dot(X.T)-y,axis=1), SIGMAS * np.sqrt(n/2)  )  )   )

'''
