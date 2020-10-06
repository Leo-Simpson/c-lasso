---
title: 'c-lasso - a package for constrained sparse and robust regression and classification in Python'
tags:
  - Python
  - Linear regression
  - Optimization
authors:
  - name: Léo Simpson
    affiliation: 1
  - name: Patrick L. Combettes
    affiliation: 2
  - name: Christian L. Müller
    affiliation: 3
affiliations:
 - name: TUM  
   index: 1
 - name: NC State
   index: 2
 - name: LMU
   index: 3
date: 13 August 2020
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:

---

# Summary

This article illustrates c-lasso, a Python package that enables sparse and robust linear
regression and classification with linear equality constraints. 


The forward model is assumed to be:

$$
y = X \beta + \sigma \epsilon \qquad \textrm{s.t.} \qquad C\beta=0
$$

Here, $X$ and $y$ are given outcome and predictor data. The vector y can be continuous (for regression) or binary (for classification). $C$ is a general constraint matrix. The vector $\beta$ comprises the unknown coefficients and $\sigma$ an unknown scale. The typical use case is logarithmic regression with compositional data, that impose the constraint $\sum_{i=1}^d \beta_i = 0$, hence $C = \mathbb{1}_d^T$.

# Statement of need 

The package handles several estimators for inferring location and scale, including the constrained Lasso, the constrained scaled Lasso, and sparse Huber M-estimation with linear equality constraints Several algorithmic strategies, including path and proximal splitting algorithms, are implemented to solve the underlying convex optimization problems. We also include two model selection strategies for determining the sparsity of the model parameters: k-fold cross-validation and stability selection. This package is intended to fill the gap between popular python tools such as `scikit-learn` which <em>cannot</em> solve sparse constrained problems and general-purpose optimization solvers such as `cvx` that do not scale well for the considered problems or are inaccurate. We show several use cases of the package, including an application of sparse log-contrast regression tasks for compositional microbiome data. We also highlight the seamless integration of the solver into `R` via the `reticulate` package. 


# Functionality

c-lasso is available on pip. You can install the package
in the shell using

```shell
pip install c_lasso
```

Here is the typical syntax to use c-lasso on python. 

```python
# let us import the main class of the package
from classo import classo_problem

# let's define a c-lasso problem instance with default setting
problem  = classo_problem(X,y,C)

# insert here possible modifications of the problem instance 

# let's solve our problem instance
problem.solve()

# finally one can visualize the instance we just solved and see solution plots as well
print(problem)
print(problem.solution)
```

## Formulations {#formulations}

Depending on the prior on the solution $\beta, \sigma$ and on the noise $\epsilon$, the previous forward model can lead to different types of estimation problems. 

Our package can solve six of those : four regression-type and two classification-type formulations. Here is an overview of those formulation, with a code snippet to see how to use those in the python. 


### *R1* Standard constrained Lasso regression: {#R1}           

$$
    \arg \min_{\beta \in \mathbb{R}^d} \left\lVert X\beta - y \right\rVert^2 + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$

This is the standard Lasso problem with linear equality constraints on the $\beta$ vector. 
The objective function combines Least-Squares for model fitting with $L_1$ penalty for sparsity.   

```python
# formulation R1
problem.formulation.huber = False
problem.formulation.concomitant = False
problem.formulation.classification = False
```

 

### *R2* Contrained sparse Huber regression: {#R2}                  

$$
    \arg \min_{\beta \in \mathbb{R}^d} h_{\rho} (X\beta - y) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$

This regression problem uses the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) as objective function 
for robust model fitting with $L_1$ and linear equality constraints on the $\beta$ vector. The parameter $\rho$ is set to $1.345$ by default [@Aigner:1976]

```python
# formulation R2
problem.formulation.huber = True
problem.formulation.concomitant = False
problem.formulation.classification = False
```

### *R3* Contrained scaled Lasso regression: {#R3}

$$
    \arg \min_{\beta \in \mathbb{R}^d} \frac{\left\lVert X\beta - y \right\rVert^2}{\sigma} + \frac{n}{2} \sigma + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$



This formulation is similar to *R1* but allows for joint estimation of the (constrained) $\beta$ vector and 
the standard deviation $\sigma$ in a concomitant fashion [@Combettes:2020.1; @Combettes:2020.2].
This is the default problem formulation in c-lasso.

```python
# formulation R3
problem.formulation.huber = False
problem.formulation.concomitant = True
problem.formulation.classification = False
```

### *R4* Contrained sparse Huber regression with concomitant scale estimation: {#R4}       

$$
    \arg \min_{\beta \in \mathbb{R}^d} \left( h_{\rho} \left( \frac{X\beta - y}{\sigma} \right) + n \right) \sigma + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$

This formulation combines *R2* and *R3* to allow robust joint estimation of the (constrained) $\beta$ vector and 
the scale $\sigma$ in a concomitant fashion [@Combettes:2020.1; @Combettes:2020.2].

```python
# formulation R4
problem.formulation.huber = True
problem.formulation.concomitant = True
problem.formulation.classification = False
```

### *C1* Contrained sparse classification with Square Hinge loss: {#C1}

$$
    \arg \min_{\beta \in \mathbb{R}^d} L(y^T X\beta - y) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$

where $L \left((r_1,...,r_n)^T \right) := \sum_{i=1}^n l(r_i)$ and $l$ is defined as :

$$
l(r) = \begin{cases} (1-r)^2 & if \quad r \leq 1 \\ 0 &if \quad r \geq 1 \end{cases}
$$

This formulation is similar to *R1* but adapted for classification tasks using the Square Hinge loss with (constrained) sparse $\beta$ vector estimation [@Lee:2013].

```python
# formulation C1
problem.formulation.huber = False
problem.formulation.concomitant = False
problem.formulation.classification = True
```

- ***C2* Contrained sparse classification with Huberized Square Hinge loss** : {#C2}       

$$
    \arg \min_{\beta \in \mathbb{R}^d} L_{\rho}(y^T X\beta - y) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$

where $L_{\rho} \left((r_1,...,r_n)^T \right) := \sum_{i=1}^n l_{\rho}(r_i)$ and $l_{\rho}$ is defined as :

$$
l_{\rho}(r) = \begin{cases} (1-r)^2 &if \quad \rho \leq r \leq 1 \\ (1-\rho)(1+\rho-2r) & if \quad r \leq \rho \\ 0 &if \quad r \geq 1 \end{cases}
$$

This formulation is similar to *C1* but uses the Huberized Square Hinge loss for robust classification with (constrained) sparse $\beta$ vector estimation [@Rosset:2007].

```python
# formulation C2
problem.formulation.huber = True
problem.formulation.concomitant = False
problem.formulation.classification = True
```



## Optimization schemes {#method}

The available problem formulations *R1-C2* require different algorithmic strategies for efficiently solving the underlying optimization problem. We have implemented four algorithms (with provable convergence guarantees) that vary in generality and are not necessarily applicable to all problems. For Huber regression, in most of the cases, mean-shift formulation of the problem is used [@Mishra:2019]. 


|             |[Path-Alg](#Path-Alg)| [DR](#DR) | [P-PDS](#P-PDS) | [PF-PDS](#PF-PDS) |
|-|:-:|:-:|:-:|:-:|
| [*R1*](#R1) | recommanded for high $\lambda$ or when path computation is require | recommanded for small $\lambda$ | possible | recommanded for complex constraints |
| [*R2*](#R2) | recommanded for high $\lambda$ or when path computation is require | recommanded for small $\lambda$ | possible | recommanded for complex constraints |
| [*R3*](#R3) | recommanded for high $\lambda$ or when path computation is require | recommanded for small $\lambda$ |  |   |
| [*R4*](#R4) |   | recommanded |   |   |
| [*C1*](#C1) | recommanded |   |   |   |
| [*C2*](#C2) | recommanded |   |   |   |



### Path algorithms (*Path-Alg*) : {#Path-Alg}
The algorithm uses the fact that the solution path along &lambda; is piecewise-affine as shown, in [@Gaines:2018].

  ```python
  problem.numerical_method = "Path-Alg"
  ```

### Douglas-Rachford-type splitting method (*DR*) : {#DR}
This algorithm is based on Doulgas Rachford splitting in a higher-dimensional product space. It makes use of the proximity operators of the perspective of the LS objective [@Combettes:2020.1; @Combettes:2020.2].

  ```python
  problem.numerical_method = "DR"
  ```

### Projected primal-dual splitting method (*P-PDS*) : {#P-PDS}
This algorithm is derived from [@Briceno:2020] and belongs to the class of proximal splitting algorithms. 

  ```python
  problem.numerical_method = "P-PDS"
  ```

### Projection-free primal-dual splitting method (*PF-PDS*) : {#PF-PDS}
This algorithm is a special case of an algorithm proposed in [@Combettes:2011] (Eq.4.5) and also belongs to the class of 
proximal splitting algorithms. 

  ```python
  problem.numerical_method = "PF-PDS"
  ```





## Model selections {#model}

Different models are implemented together with the optimization schemes, to overcome the difficulty of choosing the penalization free parameter $\lambda$. When using the package, several of those model selection can be computed with the same problem-instance.

- *Fixed Lambda* : This approach is simply letting the user choose the parameter $\lambda$, or to choose $l \in [0,1]$ such that $\lambda = l\times \lambda_{\max}$. 
The default value is a scale-dependent tuning parameter that has been proposed in [Combettes:2020.2] and derived in [@Shi:2016].
  ```python
  problem.model_selection.LAMfixed = True
  ```

- *Path Computation* :The package also leaves the possibility to us to compute the solution for a range of $\lambda$ parameters in an interval $[\lambda_{\min}, \lambda_{\max}]$. It can be done using *Path-Alg* or warm-start with any other optimization scheme. 
  ```python
  problem.model_selection.PATH = True
  ```

[comment]: <> (This can be done much faster than by computing separately the solution for each $\lambda$ of the grid, by using the Path-alg algorithm. One can also use warm starts : starting with $\beta_0 = 0$ for $\lambda_0 = \lambda_{\max}$, and then iteratvely compute $\beta_{k+1}$ using one of the optimization schemes with $\lambda = \lambda_{k+1} := \lambda_{k} - \epsilon$ and with a warm start set to $\beta_{k}$. )

- *Cross Validation* : Then one can use a model selection, to choose the appropriate penalisation. This can be done by using k-fold cross validation to find the best $\lambda \in [\lambda_{\min}, \lambda_{\max}]$ with or without "one-standard-error rule" [@Hastie:2009].
  ```python
  problem.model_selection.CV = True
  ```

- *Stability Selection* : Another variable selection model than can be used is stability selection [@Lin:2014; @Meinshausen:2010; Combettes:2020.2].
  ```python
  problem.model_selection.StabSel = True
  ```




# Example on synthetic data

#### Generate random data
The c-lasso package includes
the routine ```random_data``` that allows you to generate problem instances using normally distributed data.

```python
n,d,d_nonzero,k,sigma =100,100,5,1,0.5
(X,C,y),sol = random_data(n,d,d_nonzero,k,sigma,zerosum=True, seed = 123 )
```
This code snippet generates randomly the vectors $\beta \in R^d$ , $X \in R^{n\times d}$ , $C \in R^{k\times d}$ (here it is the all-one vector instead because of the input ```zerosum```), and $y \in R^n$ normally distributed with respect to the model $C\beta=0$, $y-X\beta \sim N(0,I_n\sigma^2)$ and $\beta$ has only d_nonzero non-null componant.


Let us now use c-lasso with this dataset. 

```python
# let's define a c-lasso problem instance with default setting
problem  = classo_problem(X,y,C)


# let's change the formulation of the problem
problem.formulation.huber  = True
problem.formulation.concomitant = False
problem.formulation.rho = 1.5


# let's add a computation of beta for a fixed lambda 
problem.model_selection.LAMfixed = True
# and set it to to 0.1*lambdamax
problem.model_selection.LAMfixedparameters.rescaled_lam = True
problem.model_selection.LAMfixedparameters.lam = 0.1

# let's add a computation of the lambda-path
problem.model_selection.PATH = True


# let's solve our problem instance
problem.solve()
```

Here, we have modified the [formulation](#formulations) of the problem in order to use [*R2*](#R2), with $\rho=1.5$. 

Then we have chosen the [model selections](#model) we want to compute : *Fixed Lambda* with $\lambda = 0.1\lambda_{\max}$ ; *Path computation* and *Stability Selection* which is computed by default. 

Finally, those problems are solved using the method `solve` which computes everything. 

#### Visualize the result 
One can, before or after having solve the problem, plot the main caracteristics of the problem solved and of its solution: 

```python
>>> # let's visualize the main parameters set in the problem instance
>>> problem

FORMULATION: R2
 
MODEL SELECTION COMPUTED:  
     Lambda fixed
     Path
     Stability selection
 
LAMBDA FIXED PARAMETERS: 
     numerical_method = DR
     rescaled lam : True
     threshold = 0.177
     lam = 0.1
     theoretical_lam = 0.1994
 
PATH PARAMETERS: 
     numerical_method : Path-Alg
     Npath = 40
     lamin = 0.013
     lamax = 1.0
 
STABILITY SELECTION PARAMETERS: 
     numerical_method : Path-Alg
     method : first
     B = 50
     q = 10
     percent_nS = 0.5
     threshold = 0.7
     lamin = 0.01
     Nlam = 50

>>> # let's the solutions found
>>> problem.solution

 LAMBDA FIXED : 
   Selected variables :  43    47    74    79    84    
   Running time :  0.094 s

 PATH COMPUTATION : 
   Running time :  0.221 s

 STABILITY SELECTION : 
   Selected variables :  43    47    74    79    84    
   Running time :  2.468 s

```

![Graphics plotted after calling problem.solution ](figures/_figure-concat.png)



As this variable selection has been computed for generated data, one can plot the real relevant variables :

```python
>>> print( list(numpy.nonzero(sol)) )
[43, 47, 74, 79, 84]
```

It is indeed the variables that have been selected with the solution threshold for a fixed lambda, and with stability selection. Let us also note that the running time is still very low in our example. 

Those remarks are comforting, but not surprising because in this example the noise is little and the number of variable is still small. 



# Acknowledgements

This work was supported by the Flatiron Institute and the Helmholtz Zentrum Munchen. 

# References



