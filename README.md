# C-lasso, a Python package for sparse regression with linear equality constraints

## Table of Contents

* [How to use the package](#how-to-use-the-package)
* [Different type of problem](#different-type-of-problem)
* [Different methods for solving the problems](#different-methods-for-solving-the-problems)
* [Two main functions](#two-main-functions)
* [Little functions](#little-functions)
* [Example](#example)


##  How to use the package

#### To install the package : 
```shell
pip install c_lasso
```

#### To import the package :
```python
from classo import *
```
#### To import the required packages  :
```shell
pip install numpy
pip install matplotlib
pip install scipy
pip install pandas
pip install time
```
    

##  Different type of problem
#### Least square :             

<img src="https://latex.codecogs.com/gif.latex?\min&space;||&space;Ax-y&space;||^2&space;&plus;&space;\lambda&space;||x||_1" />

#### Huber  :                   

<img src="https://latex.codecogs.com/gif.latex?\min&space;h_{\rho}(Ax-y)&space;&plus;&space;\lambda&space;||x||_1"  />

#### Concomitant Least square : 

<img src="https://latex.codecogs.com/gif.latex?\min&space;\frac{||&space;Ax-y&space;||^2}{\sigma}&plus;&space;n\sigma&space;&plus;&space;\lambda&space;||x||_1"  />

#### Concomitant Huber :        

<img src="https://latex.codecogs.com/gif.latex?\min&space;h_{\rho}(\frac{Ax-y}{\sigma}&space;)&space;&plus;&space;n\sigma&space;&plus;&space;\lambda&space;||x||_1" />



## Different methods for solving the problems

### Four main methods have been implemented for those.


#### Forward Backward splitting method:
Standard way to solve a convex minimisation problem with an addition of
smooth and non-smooth function : Projected Proximal Gradient Descent. This
method only works with the two non concomitants problems. For the huber
problem, we use the second formulation.

#### No-proj method
Similar to the Projected Proximal Gradient Descent, but which does not involve
a projection, which can be difficult to compute for some matrix C. Only for
non concomitant problems.

#### Double prox method
Use of Doulgas Rachford splitting algorithm which use the proximal operator of
both functions. It also solves concomitant problems, but it is usefull even in the
non concomitant case because it is usually more efficient than forward backward
splitting method. For the huber problem, we use the second formulation, then
we change it into a Least square problem of dimension m (m + d) instead of m d.

#### ODE method  
From the KKT conditions, we can derive an simple ODE for the solution of
the non concomitants problems, which shows that the solution is piecewise-
affine. For the least square, as the problem can always be reported to a a non
concomitant problem for another lambda, one can use the whole non-concomitant-
path computed with the ODE method to then solve the concomitant-path.



## Example

Here is an example of use of one of the methods  : concomitant algorithm with theoritical lambda, tested on data generated randomly. 

To generate the data :
```python
m,d,d_nonzero,k,sigma =100,100,5,1,0.5
(X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True)
```
Use of the package with default settings (example1) :
```python
problem = classo_problem(X,y,C) 
problem.solve()
print(problem.solution)
```

Results : 
```
Cross Validation         : 'not computed'
Stability Selection      :  Running time for Stability Selection    : 4.323s
Solution for a fixed lam : 'not computed'
```

![Ex1.1](figures_example1/Figure1.png)

![Ex1.2](figures_example1/Figure2.png)

![Ex1.3](figures_example1/Figure3.png)


Example of different settings (example2) : 
```python
problem                                     = classo_problem(X,y,C)
problem.formulation.huber                   = True
problem.formulation.concomitant             = False
problem.model_selection.CV                  = True
problem.model_selection.LAMfixed            = True
problem.model_selection.SSparameters.method = 'max'
problem.solve()
print(problem.solution)

problem.solution.CV.graphic(mse_max = 1.)
```

Results : 
```
Cross Validation         :  Running time for Cross Validation    : 1.878s
Stability Selection      :  Running time for Stability Selection    : 2.843s
Solution for a fixed lam :  Running time for LAM fixed       : 0.11s
```


![Ex2.1](figures_example2/Figure1.png)

![Ex2.2](figures_example2/Figure2.png)

![Ex2.3](figures_example2/Figure3.png)

![Ex2.4](figures_example2/Figure4.png)

![Ex2.5](figures_example2/Figure5.png)



