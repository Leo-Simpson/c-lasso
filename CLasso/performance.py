from CLasso.solver import classo_problem, classo_data
from CLasso.little_functions import random_data

def performance(problem,N=50,m=100,d=200,d_nonzero=5,k=3,sigma=0.5):
    problem.model_selection.SS          = True
    problem.model_selection.CV          = True
    problem.model_selection.LAMfixed    = True
    CVsuccess, SSsuccess, LAMsuccess = 0., 0., 0.
    for j in range(N):
        (X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True)
        real_support = sol != 0.
        problem.data = classo_data(X,y,C)
        problem.solve()
        CVsupport = problem.solution.CV.selected_param
        SSsupport = problem.solution.SS.selected_param
        LAMsupport = problem.solution.LAMfixed.selected_param

        CVerr = sum(CVsupport != real_support )
        SSerr = sum(SSsupport != real_support )
        LAMerr= sum(LAMsupport != real_support )

        if (CVerr ==0.) : CVsuccess +=1
        if (SSerr ==0.) : SSsuccess +=1
        if (LAMerr ==0.) : LAMsuccess +=1
    print("Success of CV  : ", round(100*CVsuccess/N,1), '%')
    print("Success of SS  : ", round(100*SSsuccess/N,1), '%')
    print("Success of LAM : ", round(100*LAMsuccess/N,1), '%')