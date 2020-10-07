---
title: 'c-lasso - a package for constrained sparse and robust regression and classification in Python'
tags:
  - Python
  - regression
  - classification
  - constrained regression
  - Lasso
  - Huber function
  - convex optimization
authors:
  - name: Léo Simpson
    affiliation: 1
  - name: Patrick L. Combettes
    affiliation: 2
  - name: Christian L. Müller
    affiliation: 3,4,5
affiliations:
 - name: Technische Universität München 
   index: 1
 - name: Department of Mathematics, North Carolina State University
   index: 2
 - name: Center for Computational Mathematics, Flatiron Institute
   index: 3
 - name: Institute of Computational Biology, Helmholtz Zentrum München
   index: 4
 - name: Department of Statistics, Ludwig-Maximilians-Universität München
   index: 5
   
date: 01 October 2020
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:

---

# Summary

We introduce `c-lasso`, a Python package that enables sparse and robust linear
regression and classification with linear equality constraints. 


The forward model is assumed to be:

$$
y = X \beta + \sigma \epsilon \qquad \textrm{s.t.} \qquad C\beta=0
$$

Here, $X \in R^{n\times d}$ is a given design matrix and the vector $y \in R^{n}$ is a continuous or binary response vector. The matrix $C$ is a general constraint matrix. The vector $\beta \in R^{d}$ contains the unknown coefficients and $\sigma$ an unknown scale. Prominent use cases are (sparse) log-contrast regression with compositional data $X$ which leads to the constraint $\sum_{i=1}^d \beta_i = 0$, i.e., $C = 1_d^T$ [@Aitchison:1984] and Generalized lasso-type
problems (see Example 3 in [James et al.](http://faculty.marshall.usc.edu/gareth-james/Research/PAC.pdf))

# Statement of need 

The package handles several estimators for inferring unknown coefficients and scale, including the constrained Lasso, the constrained scaled Lasso, and sparse Huber M-estimators with linear equality constraints, none of which can be currently solved in Python in an efficient manner. Several algorithmic strategies, including path and proximal splitting algorithms, are implemented to solve the underlying convex optimization problems at fixed regularization and across an entire regularization path. We include three model selection strategies for determining model parameter regularization levels: a theoretically derived fixed regularization, k-fold cross-validation, and stability selection. The c-lasso package is intended to fill the gap between popular python tools such as `scikit-learn` which <em>cannot</em> solve sparse constrained problems and general-purpose optimization solvers such as `cvx` that do not scale well for the considered problems and/or are inaccurate. We show several use cases of the package, including an application of sparse log-contrast regression tasks for compositional microbiome data. We also highlight the seamless integration of the solver into `R` via the `reticulate` package. 


# Functionalities

c-lasso is available on pip. You can install the package
in the shell using

```shell
pip install c_lasso
```

Here is the typical syntax to use c-lasso on python. 

```python
# to import the main class of the package
from classo import classo_problem

# to define a c-lasso problem instance with default setting
problem  = classo_problem(X,y,C)

# insert here possible modifications of the problem instance 

# to solve our problem instance
problem.solve()

# finally one can visualize the instance we just solved and see solution plots as well
print(problem)
print(problem.solution)
```

## Formulations {#formulations}

Depending on the type of the data and prior assumptions on the data, the noise $\epsilon$, and the model parameters, different types of estimation problems can be formulated. The package allows solving four regression-type and two classification-type formulations:


### *R1* Standard constrained Lasso regression: {#R1}           

$$
    \arg \min_{\beta \in \mathbb{R}^d} \left\lVert X\beta - y \right\rVert^2 + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$

This is the standard Lasso problem with linear equality constraints on the $\beta$ vector. 
The objective function combines Least-Squares (LS) for model fitting with the $L_1$-norm penalty for sparsity.   

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
for robust model fitting with an $L_1$ penalty and linear equality constraints on the $\beta$ vector. The default parameter $\rho$ is set to $1.345$ [@Huber:1981].

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

This formulation is the default problem formulation in c-lasso. It is similar to *R1* but allows for joint estimation of the (constrained) $\beta$ vector and 
the standard deviation $\sigma$ in a concomitant fashion [@Combettes:2020; @Muller:2020].

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

This formulation combines *R2* and *R3* allowing robust joint estimation of the (constrained) $\beta$ vector and 
the scale $\sigma$ in a concomitant fashion [@Combettes:2020; @Muller:2020].

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

### *C2* Contrained sparse classification with Huberized Square Hinge loss : {#C2}       

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

The available problem formulations *R1-C2* require different algorithmic strategies for efficiently solving the underlying optimization problems. We have implemented four published algorithms (with provable convergence guarantees). The c-lasso package also includes novel algorithmic extensions to solve Huber-type problems efficiently using the mean-shift formulation [@Mishra:2019]. 

Here is a summary of which algorithm can be used for each problem : 

|             |*Path-Alg*| *DR* | *P-PDS* | *PF-PDS* |
|-|:-:|:-:|:-:|:-:|
| [*R1*](#R1) | recommended for large $\lambda$ and path computation | recommended for small $\lambda$ | possible | recommended for complex constraints |
| [*R2*](#R2) | recommended for large $\lambda$ and path computation | recommended for small $\lambda$ | possible | recommended for complex constraints |
| [*R3*](#R3) | recommended for large $\lambda$ or when path computation is require | recommended for small $\lambda$ |          |                   |
| [*R4*](#R4) |                                                                    | recommended (only option)                |          |   |
| [*C1*](#C1) | recommended (only option)                                                       |                                 |          |   |
| [*C2*](#C2) | recommended (only option)                                                     |                                 |          |   |



The python syntax to use an algorithm different than recommanded is the following : 
```python
problem.numerical_method = "Path-Alg" 
# alternative options: "DR", "P-PDS", and "PF-PDS" 
```

- Path algorithms (*Path-Alg*):
The algorithm uses the fact that the solution path along &lambda; is piecewise-affine as shown in [@Gaines:2018].

- Douglas-Rachford-type splitting method (*DR*):
This algorithm is based on the Doulgas-Rachford algorithm in a higher-dimensional product space [@Combettes:2020; @Muller:2020].

- Projected primal-dual splitting method (*P-PDS*): This algorithm is derived from [@Briceno:2020] and belongs to the class of proximal splitting algorithms. 

- Projection-free primal-dual splitting method (*PF-PDS*): This algorithm is a special case of an algorithm proposed in [@Combettes:2011] (Eq.4.5) and belongs to the class of proximal splitting algorithms. 


## Model selections {#model}

Different models are implemented together with the optimization schemes, to overcome the difficulty of choosing the penalization free parameter $\lambda$. When using the package, several of those model selection can be computed with the same problem-instance.

The python syntax to use some specific model selection is the following
```python
# to perform Cross-Validation and Path computation :
problem.model_selection.LAMfixed = False
problem.model_selection.PATH = True
problem.model_selection.CV = True
problem.model_selection.StabSel = False
# obviously any other combination also works
```


- *Fixed Lambda* : This approach is simply letting the user choose the parameter $\lambda$, or to choose $l \in [0,1]$ such that $\lambda = l\times \lambda_{\max}$. 
The default value is a scale-dependent tuning parameter that has been proposed in [Muller:2020] and derived in [@Shi:2016].

- *Path Computation* :The package also leaves the possibility to us to compute the solution for a range of $\lambda$ parameters in an interval $[\lambda_{\min}, \lambda_{\max}]$. It can be done using *Path-Alg* or warm-start with any other optimization scheme. 

[comment]: <> (This can be done much faster than by computing separately the solution for each $\lambda$ of the grid, by using the Path-alg algorithm. One can also use warm starts : starting with $\beta_0 = 0$ for $\lambda_0 = \lambda_{\max}$, and then iteratvely compute $\beta_{k+1}$ using one of the optimization schemes with $\lambda = \lambda_{k+1} := \lambda_{k} - \epsilon$ and with a warm start set to $\beta_{k}$. )

- *Cross Validation* : Then one can use a model selection, to choose the appropriate penalisation. This can be done by using k-fold cross validation to find the best $\lambda \in [\lambda_{\min}, \lambda_{\max}]$ with or without "one-standard-error rule" [@Hastie:2009].


- *Stability Selection* : Another variable selection model than can be used is stability selection [@Lin:2014; @Meinshausen:2010; Muller:2020].



## Example on R 

As an alternative, one can use this package in R instead of python by calling the python package with the Rlibrary ```reticulate```. As an example, this code snippet used in R will perform regression with a fixed lambda set to $\lambda = 0.1\lambda_{\max}$.

One should be careful with the inputs : X should be a ```matrix```, C as well, but y should be an ```array``` (if one set y to be matrix $1\times n$ for example, c-lasso will not work).

```r
problem<- classo$classo_problem(X=X,C=C,y=array(y))
problem$model_selection$LAMfixed <- TRUE
problem$model_selection$StabSel <- FALSE
problem$model_selection$LAMfixedparameters$rescaled_lam <- TRUE
problem$model_selection$LAMfixedparameters$lam <- 0.1
problem$solve()

# extract outputs
beta <- as.matrix(map_dfc(problem$solution$LAMfixed$beta, as.numeric))
```


# Example on synthetic data


The c-lasso package includes
the routine ```random_data``` that allows you to generate problem instances using normally distributed data.

[comment]: <> (It generates randomly the vectors $\beta \in R^d$ , $X \in R^{n\times d}$, $C \in R^{k\times d}$ [can also be the all-one vector with the input ```zerosum``` set to true], and $y \in R^n$ normally distributed with respect to the model $C\beta=0$, $y-X\beta \sim N[0,I_n\sigma^2]$ and $\beta$ has only d_nonzero non-null componant. )


 It allows perform some functionality of the package on synthetic data as an example. 

```python
from classo import classo_problem, random_data

n,d,d_nonzero,k,sigma =100,100,5,1,0.5
(X,C,y),sol = random_data(n,d,d_nonzero,k,sigma,zerosum=True, seed = 123 )
print("Relevant variables  : {}".format(list(numpy.nonzero(sol)) ) )

problem  = classo_problem(X,y,C)

problem.formulation.huber  = True
problem.formulation.concomitant = False
problem.formulation.rho = 1.5

problem.model_selection.LAMfixed = True
problem.model_selection.PATH = True
problem.model_selection.LAMfixedparameters.rescaled_lam = True
problem.model_selection.LAMfixedparameters.lam = 0.1

problem.solve()

print(problem.solution)
```

The [formulation](#formulations) used is [*R2*](#R2), with $\rho=1.5$. The [model selections](#model) used are *Fixed Lambda* with $\lambda = 0.1\lambda_{\max}$ , *Path computation* and *Stability Selection* which is computed by default. 

The output is then : 

```shell
Relevant variables  : [43 47 74 79 84]

 LAMBDA FIXED : 
   Selected variables :  43    47    74    79    84    
   Running time :  0.294s

 PATH COMPUTATION : 
   Running time :  0.566s

 STABILITY SELECTION : 
   Selected variables :  43    47    74    79    84    
   Running time :  5.3s
```

![Graphics plotted after calling problem.solution ](figures/_figure-concat.png)


It is indeed the variables that have been selected with the solution threshold for a fixed lambda, and with stability selection. Let us also note that the running time is still very low in our example. Those remarks are comforting, but not surprising because in this example the noise is little and the number of variable is still small. 


# Acknowledgements

This work was supported by the Flatiron Institute and the Helmholtz Zentrum Munchen. 

# References



