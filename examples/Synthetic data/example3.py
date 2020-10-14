from CLasso import *
import matplotlib.pyplot as plt
m,d,d_nonzero,k,sigma =100,100,5,1,0.5
(X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True, seed = 2, classification = True)
problem = classo_problem(X,y,C)
problem.formulation.classification = True
problem.formulation.huber = True
problem.solve()
print(problem)
plt.bar(range(len(sol)),sol),   plt.title("Real Solution"),   plt.show()
print(problem.solution)
