from CLasso import *
m,d,d_nonzero,k,sigma =100,90,5,1,0.5
(X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True,seed=4)
problem = classo_problem(X,y,C)
problem.solve()
print(problem)
print(problem.solution)


