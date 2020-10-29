<img src="https://i.imgur.com/2nGwlux.png" alt="c-lasso" height="150" align="right"/>

# c-lasso: a Python package for constrained sparse regression and classification 
=========


c-lasso is a Python package that enables sparse and robust linear regression and classification with linear equality
constraints on the model parameters. For detailed info, one can check the [documentation](https://c-lasso.readthedocs.io/en/latest/).

The forward model is assumed to be: 

<img src="https://latex.codecogs.com/gif.latex?y=X\beta&plus;\sigma\epsilon\qquad\text{s.t.}\qquad&space;C\beta=0" title="y=X\beta+\sigma\epsilon\qquad\text{s.t.}\qquad C\beta=0" />

Here, y and X are given outcome and predictor data. The vector y can be continuous (for regression) or binary (for classification). C is a general constraint matrix. The vector &beta; comprises the unknown coefficients and &sigma; an 
unknown scale.

The package handles several different estimators for inferring &beta; (and &sigma;), including 
the constrained Lasso, the constrained scaled Lasso, and sparse Huber M-estimation with linear equality constraints.
Several different algorithmic strategies, including path and proximal splitting algorithms, are implemented to solve 
the underlying convex optimization problems.

We also include two model selection strategies for determining the sparsity of the model parameters: k-fold cross-validation and stability selection.   

This package is intended to fill the gap between popular python tools such as [scikit-learn](https://scikit-learn.org/stable/) which CANNOT solve sparse constrained problems and general-purpose optimization solvers that do not scale well for the considered problems.

Below we show several use cases of the package, including an application of sparse *log-contrast*
regression tasks for *compositional* microbiome data.

The code builds on results from several papers which can be found in the [References](#references).

## Table of Contents

* [Installation](#installation)
* [Regression and classification problems](#regression-and-classification-problems)
* [Getting started](#getting-started)
* [Log-contrast regression for microbiome data](#log-contrast-regression-for-microbiome-data)
* [Optimization schemes](#optimization-schemes)
* [Structure of the code](#structure-of-the-code)


* [References](#references)


##  Installation

c-lasso is available on pip. You can install the package
in the shell using

```shell
pip install c-lasso
```
To use the c-lasso package in Python, type 

```python
from classo import *
```

The `c-lasso` package depends on the following Python packages:

`numpy` ; 
`matplotlib` ; 
`scipy` ; 
`pandas` ; 
`h5py` .

##  Regression and classification problems

The c-lasso package can solve six different types of estimation problems: 
four regression-type and two classification-type formulations.

#### [R1] Standard constrained Lasso regression:             

<img src="https://latex.codecogs.com/gif.latex?\arg\min_{\beta\in&space;R^d}&space;||&space;X\beta-y&space;||^2&space;&plus;&space;\lambda&space;||\beta||_1&space;\qquad\mbox{s.t.}\qquad&space;C\beta=0" />

This is the standard Lasso problem with linear equality constraints on the &beta; vector. 
The objective function combines Least-Squares for model fitting with l1 penalty for sparsity.   

#### [R2] Contrained sparse Huber regression:                   

<img src="https://latex.codecogs.com/gif.latex?\arg\min_{\beta\in&space;R^d}&space;h_{\rho}(X\beta-y&space;)&space;&plus;&space;\lambda&space;||\beta||_1&space;\qquad\mbox{s.t.}\qquad&space;C\beta=0" />

This regression problem uses the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) as objective function 
for robust model fitting with l1 and linear equality constraints on the &beta; vector. The parameter &rho;=1.345.

#### [R3] Contrained scaled Lasso regression: 

<img src="https://latex.codecogs.com/gif.latex?\arg&space;\min_{\beta&space;\in&space;\mathbb{R}^d,&space;\sigma&space;>&space;0}&space;\frac{||&space;X\beta&space;-&space;y||^2}{\sigma}&space;&plus;&space;\frac{n}{2}&space;\sigma&plus;&space;\lambda&space;||\beta||_1&space;\qquad&space;\mbox{s.t.}&space;\qquad&space;C\beta&space;=&space;0" title="\arg \min_{\beta \in \mathbb{R}^d, \sigma > 0} \frac{|| X\beta - y||^2}{\sigma} + \frac{n}{2} \sigma+ \lambda ||\beta||_1 \qquad \mbox{s.t.} \qquad C\beta = 0" />

This formulation is similar to [R1] but allows for joint estimation of the (constrained) &beta; vector and 
the standard deviation &sigma; in a concomitant fashion (see [References](#references) [4,5] for further info).
This is the default problem formulation in c-lasso.

#### [R4] Contrained sparse Huber regression with concomitant scale estimation:        

<img src="https://latex.codecogs.com/gif.latex?\arg&space;\min_{\beta&space;\in&space;\mathbb{R}^d,&space;\sigma&space;>&space;0}&space;\left(&space;h_{\rho}&space;\left(&space;\frac{&space;X\beta&space;-&space;y}{\sigma}&space;\right)&plus;&space;n&space;\right)&space;\sigma&plus;&space;\lambda&space;||\beta||_1&space;\qquad&space;\mbox{s.t.}&space;\qquad&space;C\beta&space;=&space;0" title="\arg \min_{\beta \in \mathbb{R}^d, \sigma > 0} \left( h_{\rho} \left( \frac{ X\beta - y}{\sigma} \right)+ n \right) \sigma+ \lambda ||\beta||_1 \qquad \mbox{s.t.} \qquad C\beta = 0" />

This formulation combines [R2] and [R3] to allow robust joint estimation of the (constrained) &beta; vector and 
the scale &sigma; in a concomitant fashion (see [References](#references) [4,5] for further info).

#### [C1] Contrained sparse classification with Square Hinge loss: 

<img src="https://latex.codecogs.com/gif.latex?\arg&space;\min_{\beta&space;\in&space;\mathbb{R}^d}&space;\sum_{i=1}^n&space;l(y_i&space;x_i^\top&space;\beta)&space;&plus;&space;\lambda&space;\left\lVert&space;\beta\right\rVert_1&space;\qquad&space;s.t.&space;\qquad&space;C\beta&space;=&space;0" title="\arg \min_{\beta \in \mathbb{R}^d} \sum_{i=1}^n l(y_i x_i \beta) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad C\beta = 0" />

where the x<sub>i</sub> are the rows of X and l is defined as:

<img src="https://latex.codecogs.com/gif.latex?l(r)&space;=&space;\begin{cases}&space;(1-r)^2&space;&&space;if&space;\quad&space;r&space;\leq&space;1&space;\\&space;0&space;&if&space;\quad&space;r&space;\geq&space;1&space;\end{cases}" title="l(r) = \begin{cases} (1-r)^2 & if \quad r \leq 1 \\ 0 &if \quad r \geq 1 \end{cases}" />

This formulation is similar to [R1] but adapted for classification tasks using the Square Hinge loss
with (constrained) sparse &beta; vector estimation.

#### [C2] Contrained sparse classification with Huberized Square Hinge loss:        

<img src="https://latex.codecogs.com/gif.latex?\arg&space;\min_{\beta&space;\in&space;\mathbb{R}^d}&space;\sum_{i=1}^n&space;l_{\rho}(y_i&space;x_i^\top\beta)&space;&plus;&space;\lambda&space;\left\lVert&space;\beta\right\rVert_1&space;\qquad&space;s.t.&space;\qquad&space;C\beta&space;=&space;0" title="\arg \min_{\beta \in \mathbb{R}^d} \sum_{i=1}^n l_{\rho}(y_i x_i\beta) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad C\beta = 0" />

where the x<sub>i</sub> are the rows of X and l<sub>ρ</sub> is defined as:

<img src="https://latex.codecogs.com/gif.latex?l_{\rho}(r)&space;=&space;\begin{cases}&space;(1-r)^2&space;&if&space;\quad&space;\rho&space;\leq&space;r&space;\leq&space;1&space;\\&space;(1-\rho)(1&plus;\rho-2r)&space;&&space;if&space;\quad&space;r&space;\leq&space;\rho&space;\\&space;0&space;&if&space;\quad&space;r&space;\geq&space;1&space;\end{cases}" title="l_{\rho}(r) = \begin{cases} (1-r)^2 &if \quad \rho \leq r \leq 1 \\ (1-\rho)(1+\rho-2r) & if \quad r \leq \rho \\ 0 &if \quad r \geq 1 \end{cases}" />


This formulation is similar to [C1] but uses the Huberized Square Hinge loss for robust classification 
with (constrained) sparse &beta; vector estimation.


## Getting started

#### Basic example

We begin with a basic example that shows how to run c-lasso on synthetic data. This example and the next one can be found on the notebook 'Synthetic data Notebook.ipynb'

The c-lasso package includes
the routine ```random_data``` that allows you to generate problem instances using normally distributed data.

```python
m,d,d_nonzero,k,sigma =100,200,5,1,0.5
(X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True,seed=1)
```
This code snippet generates a problem instance with sparse &beta; in dimension
d=100 (sparsity d_nonzero=5). The design matrix X comprises n=100 samples generated from an i.i.d standard normal
distribution. The dimension of the constraint matrix C is d x k matrix. The noise level is &sigma;=0.5. 
The input ```zerosum=True``` implies that C is the all-ones vector and C&beta;=0. The n-dimensional outcome vector y
and the regression vector &beta; is then generated to satisfy the given constraints. 

Next we can define a default c-lasso problem instance with the generated data:
```python
problem = classo_problem(X,y,C) 
```
You can look at the generated problem instance by typing:

```python
print(problem)
```

This gives you a summary of the form:

```
FORMULATION: R3
 
MODEL SELECTION COMPUTED:  
     Stability selection
 
STABILITY SELECTION PARAMETERS: 
     numerical_method : not specified
     method : first
     B = 50
     q = 10
     percent_nS = 0.5
     threshold = 0.7
     lamin = 0.01
     Nlam = 50
```
As we have not specified any problem, algorithm, or model selection settings, this problem instance
represents the *default* settings for a c-lasso instance: 
- The problem is of regression type and uses formulation [R3], i.e. with concomitant scale estimation. 
- The *default* optimization scheme is the path algorithm (see [Optimization schemes](#optimization-schemes) for further info). 
- For model selection, stability selection at a theoretically derived &lambda; value is used (see [Reference](#references) [4] for details). Stability selection comprises a relatively large number of parameters. For a description of the settings, we refer to the more advanced examples below and the API.

You can solve the corresponding c-lasso problem instance using

```python
problem.solve()
```

After completion, the results of the optimization and model selection routines 
can be visualized using

```python
print(problem.solution)
```

The command shows the running time(s) for the c-lasso problem instance, and the selected variables for sability selection

```
STABILITY SELECTION : 
   Selected variables :  1    5    14    17    18    
   Running time :  0.663s
```

Here, we only used stability selection as *default* model selection strategy. 
The command also allows you to inspect the computed stability profile for all variables 
at the theoretical &lambda; 

![1.StabSel](https://github.com/Leo-Simpson/Figures/blob/master/basic/StabSel.png)


The refitted &beta; values on the selected support are also displayed in the next plot

![beta](https://github.com/Leo-Simpson/Figures/blob/master/basic/beta.png)


#### Advanced example             

In the next example, we show how one can specify different aspects of the problem 
formulation and model selection strategy.

```python
m,d,d_nonzero,k,sigma =100,200,5,0,0.5
(X,C,y),sol = random_data(m,d,d_nonzero,k,sigma,zerosum=True,seed=4)
problem                                     = classo_problem(X,y,C)
problem.formulation.huber                   = True
problem.formulation.concomitant             = False
problem.model_selection.CV                  = True
problem.model_selection.LAMfixed            = True
problem.model_selection.PATH                = True
problem.model_selection.StabSelparameters.method = 'max'
problem.model_selection.CVparameters.seed = 1
problem.model_selection.LAMfixedparameters.rescaled_lam = True
problem.model_selection.LAMfixedparameters.lam = .1

problem.solve()
print(problem)

print(problem.solution)

```

Results : 
```
FORMULATION: R2
 
MODEL SELECTION COMPUTED:  
     Lambda fixed
     Path
     Cross Validation
     Stability selection
 
LAMBDA FIXED PARAMETERS: 
     numerical_method = Path-Alg
     rescaled lam : True
     threshold = 0.106
     lam = 0.1
     theoretical_lam = 0.224
 
PATH PARAMETERS: 
     numerical_method : Path-Alg
     lamin = 0.001
     Nlam = 80
 
 
CROSS VALIDATION PARAMETERS: 
     numerical_method : Path-Alg
     one-SE method : True
     Nsubset = 5
     lamin = 0.001
     Nlam = 80
 
 
STABILITY SELECTION PARAMETERS: 
     numerical_method : Path-Alg
     method : max
     B = 50
     q = 10
     percent_nS = 0.5
     threshold = 0.7
     lamin = 0.01
     Nlam = 50


 LAMBDA FIXED : 
   Selected variables :  17    59    76    123    137    
   Running time :  0.234s

 PATH COMPUTATION : 
   Running time :  0.557s

 CROSS VALIDATION : 
   Selected variables :  16    17    57    59    64    73    74    76    93    115    123    134    137    181    
   Running time :  1.751s

 STABILITY SELECTION : 
   Selected variables :  1    3    7    12    
   Running time :  8.391s

```






![2.StabSel](https://github.com/Leo-Simpson/Figures/blob/master/advanced/StabSel.png)

![2.StabSel-beta](https://github.com/Leo-Simpson/Figures/blob/master/advanced/StabSel-beta.png)

![2.CV-beta](https://github.com/Leo-Simpson/Figures/blob/master/advanced/CVbeta.png)

![2.CV-graph](https://github.com/Leo-Simpson/Figures/blob/master/advanced/CV.png)

![2.LAM-beta](https://github.com/Leo-Simpson/Figures/blob/master/advanced/beta.png)

![2.Path](https://github.com/Leo-Simpson/Figures/blob/master/advanced/Beta-path.png)


## Log-contrast regression for microbiome data

A couple of datasets have been studied here. One can find this analysis on the jupyter notebook "example-notebook.ipynb". Some examples taken from this notebook are presented below.

#### BMI prediction using the COMBO dataset 

Here is now the main results from the COMBO data analysis taken from the notebook "example-notebook.ipynb",
at the section " Filtered Combo data".

```python
from classo import *

# Load microbiome and covariate data X
X_C = csv_to_np('COMBO_data/CaloriData.csv',begin=0).astype(float)
X_F = csv_to_np('COMBO_data/FatData.csv',begin=0).astype(float)
X0  = csv_to_np('COMBO_data/filtered_data/GeneraFilteredCounts.csv',begin=0).astype(float)


# Load BMI measurements y
y   = csv_to_np('COMBO_data/BMI.csv',begin=0).astype(float)[:,0]

# Load genus and covariate labels
labels  = csv_to_np('COMBO_data/filtered_data/GeneraFilteredPhylo.csv').astype(str)[:,-1]


# Normalize/transform data
y   = y - np.mean(y) #BMI data (n=96)
X_C = X_C - np.mean(X_C, axis=0)  #Covariate data (Calorie)
X_F = X_F - np.mean(X_F, axis=0)  #Covariate data (Fat)
X0 = clr(X0, 1 / 2).T

# Set up design matrix and zero-sum constraints for 45 genera
X      = np.concatenate((X0, X_C, X_F, np.ones((len(X0), 1))), axis=1) # Joint microbiome and covariate data and offset
label = np.concatenate([labels,np.array(['Calorie','Fat','Bias'])])
C = np.ones((1,len(X[0])))
C[0,-1],C[0,-2],C[0,-3] = 0.,0.,0.

# Set up c-lassso problem
problem = classo_problem(X,y,C, label=label)


# Use stability selection with theoretical lambda [Combettes & Müller, 2020b]
problem.model_selection.StabSel                       = True
problem.model_selection.StabSelparameters.method      = 'lam'
problem.model_selection.StabSelparameters.seed        = 2

# Use formulation R3
problem.formulation.concomitant = True

problem.solve()
print(problem)
print(problem.solution)

# Use formulation R4
problem.formulation.huber = True
problem.formulation.concomitant = True

problem.solve()
print(problem)
print(problem.solution)

```

![3.Stability profile R3](https://github.com/Leo-Simpson/Figures/blob/master/exampleFilteredCOMBO/R3-StabSel.png)

![3.Beta solution R3](https://github.com/Leo-Simpson/Figures/blob/master/exampleFilteredCOMBO/R3-StabSel-beta.png)

![3.Stability profile R4](https://github.com/Leo-Simpson/Figures/blob/master/exampleFilteredCOMBO/R4-StabSel.png)

![3.Beta solution R4](https://github.com/Leo-Simpson/Figures/blob/master/exampleFilteredCOMBO/R4-StabSel-beta.png)



#### pH prediction using the Central Park soil dataset 

Next part of the notebook, namely, the analysis of "pH data" : 

Here are the parameters of what we compute : 
```
FORMULATION: R3
 
MODEL SELECTION COMPUTED:  
     Lambda fixed
     Path
     Stability selection
 
LAMBDA FIXED PARAMETERS: 
     numerical_method = Path-Alg
     rescaled lam : True
     threshold = 0.004
     lam : theoretical
     theoretical_lam = 0.2182
 
PATH PARAMETERS: 
     numerical_method : Path-Alg
     lamin = 0.001
     Nlam = 80
 
 
STABILITY SELECTION PARAMETERS: 
     numerical_method : Path-Alg
     method : lam
     B = 50
     q = 10
     percent_nS = 0.5
     threshold = 0.7
     lam = theoretical
     theoretical_lam = 0.3085
```

And here is the result : 

```
LAMBDA FIXED : 
   Sigma  =  0.198
   Selected variables :  14    18    19    39    43    57    62    85    93    94    104    107    
   Running time :  0.008s

 PATH COMPUTATION : 
   Running time :  0.12s

 STABILITY SELECTION : 
   Selected variables :  2    12    15    
   Running time :  0.287s
```

![Ex4.1](https://github.com/Leo-Simpson/Figures/blob/master/examplePH/R3-Beta-path.png)

![Ex4.2](https://github.com/Leo-Simpson/Figures/blob/master/examplePH/R3-Sigma-path.png)

![Ex4.3](https://github.com/Leo-Simpson/Figures/blob/master/examplePH/R3-StabSel.png)

![Ex4.4](https://github.com/Leo-Simpson/Figures/blob/master/examplePH/R3-StabSel-beta.png)

![Ex4.5](https://github.com/Leo-Simpson/Figures/blob/master/examplePH/R3-beta.png)


## Optimization schemes

The available problem formulations [R1-C2] require different algorithmic strategies for 
efficiently solving the underlying optimization problem. We have implemented four 
algorithms (with provable convergence guarantees) that vary in generality and are not 
necessarily applicable to all problems. For each problem type, c-lasso has a default algorithm 
setting that proved to be the fastest in our numerical experiments.

### Path algorithms (Path-Alg) 
This is the default algorithm for non-concomitant problems [R1,R3,C1,C2]. 
The algorithm uses the fact that the solution path along &lambda; is piecewise-
affine (as shown, e.g., in [1]). When Least-Squares is used as objective function,
we derive a novel efficient procedure that allows us to also derive the 
solution for the concomitant problem [R2] along the path with little extra computational overhead.

### Projected primal-dual splitting method (P-PDS):
This algorithm is derived from [2] and belongs to the class of 
proximal splitting algorithms. It extends the classical Forward-Backward (FB) 
(aka proximal gradient descent) algorithm to handle an additional linear equality constraint
via projection. In the absence of a linear constraint, the method reduces to FB.
This method can solve problem [R1]. For the Huber problem [R3], 
P-PDS can solve the mean-shift formulation of the problem (see [6]).

### Projection-free primal-dual splitting method (PF-PDS):
This algorithm is a special case of an algorithm proposed in [3] (Eq.4.5) and also belongs to the class of 
proximal splitting algorithms. The algorithm does not require projection operators 
which may be beneficial when C has a more complex structure. In the absence of a linear constraint, 
the method reduces to the Forward-Backward-Forward scheme. This method can solve problem [R1]. 
For the Huber problem [R3], PF-PDS can solve the mean-shift formulation of the problem (see [6]).

### Douglas-Rachford-type splitting method (DR)
This algorithm is the most general algorithm and can solve all regression problems 
[R1-R4]. It is based on Doulgas Rachford splitting in a higher-dimensional product space.
It makes use of the proximity operators of the perspective of the LS objective (see [4,5])
The Huber problem with concomitant scale [R4] is reformulated as scaled Lasso problem 
with the mean shift (see [6]) and thus solved in (n + d) dimensions. 


## Structure of the code

![Structure](https://github.com/Leo-Simpson/Figures/blob/master/classo_structure.png)


## References 

* [1] B. R. Gaines, J. Kim, and H. Zhou, [Algorithms for Fitting the Constrained Lasso](https://www.tandfonline.com/doi/abs/10.1080/10618600.2018.1473777?journalCode=ucgs20), J. Comput. Graph. Stat., vol. 27, no. 4, pp. 861–871, 2018.

* [2] L. Briceno-Arias and S.L. Rivera, [A Projected Primal–Dual Method for Solving Constrained Monotone Inclusions](https://link.springer.com/article/10.1007/s10957-018-1430-2?shared-article-renderer), J. Optim. Theory Appl., vol. 180, Issue 3, March 2019.

* [3] P. L. Combettes and J.C. Pesquet, [Primal-Dual Splitting Algorithm for Solving Inclusions with Mixtures of Composite, Lipschitzian, and Parallel-Sum Type Monotone Operators](https://arxiv.org/pdf/1107.0081.pdf), Set-Valued and Variational Analysis, vol. 20, pp. 307-330, 2012.

* [4] P. L. Combettes and C. L. Müller, [Perspective M-estimation via proximal decomposition](https://arxiv.org/abs/1805.06098), Electronic Journal of Statistics, 2020, [Journal version](https://projecteuclid.org/euclid.ejs/1578452535) 

* [5] P. L. Combettes and C. L. Müller, [Regression models for compositional data: General log-contrast formulations, proximal optimization, and microbiome data applications](https://arxiv.org/abs/1903.01050), Statistics in Bioscience, 2020.

* [6] A. Mishra and C. L. Müller, [Robust regression with compositional covariates](https://arxiv.org/abs/1909.04990), arXiv, 2019.

* [7] S. Rosset and J. Zhu, [Piecewise linear regularized solution paths](https://projecteuclid.org/euclid.aos/1185303996), Ann. Stat., vol. 35, no. 3, pp. 1012–1030, 2007.


