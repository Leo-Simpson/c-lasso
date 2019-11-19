# Constrained sparse regression functions in Python

### To install the package : 
```shell
pip install c_lasso
```

### To import the package :
```python
from classo import *
```
### To import the required packages  :
```shell
pip install numpy
pip install matplotlib
pip install scipy
pip install pandas
pip install time
```
    

##  Different type of problem : 
#### Least square :             

<img src="https://latex.codecogs.com/gif.latex?\min&space;||&space;Ax-y&space;||^2&space;&plus;&space;\lambda&space;||x||_1" />

#### Huber  :                   

<img src="https://latex.codecogs.com/gif.latex?\min&space;h_{\rho}(Ax-y)&space;&plus;&space;\lambda&space;||x||_1"  />

#### Concomitant Least square : 

<img src="https://latex.codecogs.com/gif.latex?\min&space;\frac{||&space;Ax-y&space;||^2}{\sigma}&plus;&space;n\sigma&space;&plus;&space;\lambda&space;||x||_1"  />

#### Concomitant Huber :        

<img src="https://latex.codecogs.com/gif.latex?\min&space;h_{\rho}(\frac{Ax-y}{\sigma}&space;)&space;&plus;&space;n\sigma&space;&plus;&space;\lambda&space;||x||_1" />



## Different methods for solving the problems : 

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



## Two main functions 

### For solving the problem for a fix \lambda : 
```python
fixlasso (matrix, lam, typ ='LS ', meth ='choose ', plot_time =True, plot_sol =True, plot_sigm =True, rho = 1.345)
```
  
#### matrix
Matrices (A;C; y) of the problem.

#### lam
Lambda/lambdamax in [0; 1] where lambdamax is a majoration of the lambda when the solution is null 
(depends on the type of problem).

#### typ
Type of problem : 'Huber', 'Concomitant' , 'Concomitant Huber' or 'LS'. 
Any other string will set the type of problem to Least Square.

#### meth
Method to solve the problem. If wrong input, the algorithm choose the method according to lambda
    - Possibilities for types 'LS' and 'Huber' : 'FB', 'Noproj', '2prox', 'ODE'.
    - Possibilities for type 'Concomitant' : '2prox', 'ODE'.
    - Possibilities for type 'Concomitant Huber' : '2prox'.

#### plot_time
If set to True : prints the running time.

#### plot_sol
If set to True : plots the solution in a bars diagram.

#### plot_sigm
If set to True and the type we solve a concomitant problem : prints sigma/sigmamax.

#### rho
Normalized rho for non-concomitant Huber problem : rho * sqrt(m) / norm_inf(y)
Unormalized sigma for concomitant Huber problem.


#### returns
The function returns : 
    An 'numpy.ndarray' type vector representing the solution betafor non concomitant problems, 
    A tuple containing beta and sigma for the concomitant problems.






### For solving the problem for the whole path :
```python
pathlasso (matrix, lambdas = 'choose ', lamin = 1e -2, typ= 'LS ', meth = 'ODE ', plot_time = True, plot_sol = True, plot_sigm = True, rho = 1.345, compare = False )
```


#### matrix
Matrices (A;C; y) of the problem.

#### lambdas
Gives the list of lambda/lambdamax in [0; 1] where we need the problem to be solved. 
If a boolean is given, it is the next parameter that will give the path.

#### lamin
If lambdas is a boolean, it gives the lambda/lambdamax minimum : the algorithm will solve the problem for all lambda in [lamin * lambdamax; lambdamax] (with 100 points).	

#### typ
Type of problem : 'Huber', 'Concomitant' , 'Concomitant Huber' or 'LS'. 
Any other string will set the type of problem to Least Square.

#### meth
Method to solve the problem. If wrong input, the algorithm choose the method according to lambda.
    - Possibilities for types 'LS' and 'Huber' : 'FB', 'Noproj', '2prox', 'ODE'.
    - Possibilities for type 'Concomitant' : '2prox', 'ODE'.
    - Possibilities for type 'Concomitant Huber' : '2prox'.

For each case except 'ODE', the algorithm solves the problem for each lambda of the path using warm starts.


#### plot_time
If set to True : prints the running time.

#### plot_sol
If set to True : plots the solution in a bars diagram.

#### plot_sigm
If set to True and the type we solve a concomitant problem : prints <img src="https://latex.codecogs.com/gif.latex?\sigma&space;/&space;\sigma_{max}" />.

#### rho
Normalized rho for non-concomitant Huber problem : <img src="https://latex.codecogs.com/gif.latex?\rho&space;*&space;\sqrt{m}&space;/&space;||y||_{\infty}" />
Unormalized sigma for concomitant Huber problem.


#### returns
The function returns :  
    a list 'numpy.ndarray' type vector representing the solution beta for each lambda ;
    the list of lambda corresponding (unormalized), 
    also the list of sigmas for the concomitant problems.





## Little functions :
### For computing the theoretical lambda/lambdamax in the case of concomitant problems :  
```python
model_selection(m,d)
```
Where m is the number of sample and d the number of parameter, it returns : <img src="https://latex.codecogs.com/gif.latex?lam0&space;=&space;\sqrt{2/m}&space;\Phi^{-1}(1-t)"  />, with <img src="https://latex.codecogs.com/gif.latex?\Phi^{-1}" /> the quantile function for the standard normal distribution, and t is the solution to the equation <img src="https://latex.codecogs.com/gif.latex?t.p&space;=&space;\Phi^{-1}(1-t)^4&space;&plus;&space;2\Phi^{-1}(1-t)^2"  />


### For computing the solution using cross-validation and the previous main functions : 
```python
CV(matrices,k=5,typ='LS',test=0.4,lamin=1e-2, print_lam= True)
```
Where k is the number of 'cluster' used, test is the proportion of sample kept for testing, and print lam tells us if the function also print the lambda/lambdamax used. The function returns the solution Beta as a 'numpy.ndarray'.   



## Example : 

Here is an example of use of one of the methods  : concomitant algorithm with theoritical lambda, tested on data generated randomly. 


```python
from classo import *
m,d,d_nonzero,k,sigma =100,200,5,5,0.5
matrices,sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True)

lam0 = model_selection(m,d)
plt.bar(range(len(sol)),sol),plt.title("True solution Beta-hat"),plt.savefig('True solution Beta-hat.png'),plt.show()
X1,s = Classo(matrices,lam0,typ='Concomitant')
```
Results : 
```python
sigma =  0.578
Running time : 0.04192 sec
```

![betah](figures/betah.png)


![beta](figures/beta.png)



One can also compute the solution for a lambda-path : 
```python
sol,path = pathlasso(matrices,lamin=0.05)
```

Results : 
Running time : 0.07373 sec

![path](figures/path.png)
