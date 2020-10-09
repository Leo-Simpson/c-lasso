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

We introduce `c-lasso`, a Python package that enables sparse and robust linear regression and classification with linear equality constraints. 
The underlying statistical forward model is assumed to be of the following form:

$$
y = X \beta + \sigma \epsilon \qquad \textrm{s.t.} \qquad C\beta=0
$$

Here, $X \in R^{n\times d}$ is a given design matrix and the vector $y \in R^{n}$ is a continuous or binary response vector. The matrix $C$ is a general
constraint matrix. The vector $\beta \in R^{d}$ contains the unknown coefficients and $\sigma$ an unknown scale. Prominent use cases are (sparse) log-contrast
regression with compositional data $X$, leading to the constraint $\sum_{i=1}^d \beta_i = 0$ (i.e., $C = 1_d^T$) [@Aitchison:1984] and Generalized Lasso-type
problems (see, e.g, [@James:2020], Example 3). The `c-lasso` package provides several estimators for inferring unknown coefficients and scale (i.e., perspective M-estimators [@Combettes:2020a]) of the form:

$$
    \arg \min_{\beta \in \mathbb{R}^d, \sigma \in \mathbb{R_0}} f\left(X\beta - y,{\sigma} \right) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$

This includes the constrained Lasso, the constrained scaled Lasso, and sparse Huber M-estimators with linear equality constraints.

# Statement of need 

Currently, there is no Python package available that can solve these ubiquitous statistical estimation problems in a fast and efficient manner. 
`c-lasso` provides algorithmic strategies, including path and proximal splitting algorithms, to solve the underlying convex optimization problems with provable convergence guarantees. The c-lasso package is intended to fill the gap between popular Python tools such as [`scikit-learn`](https://scikit-learn.org/stable/) which <em>cannot</em> solve these constrained problems and general-purpose optimization solvers such as [`cvxpy`](https://www.cvxpy.org) that do not scale well for these problems and/or are inaccurate. `c-lasso` can solve the estimation problems at fixed regularization level and across an entire regularization path and includes three model selection strategies for determining model parameter regularization levels: a theoretically derived fixed regularization, k-fold cross-validation, and stability selection. We show several use cases of the package, including an application of sparse log-contrast regression tasks for compositional microbiome data, and highlight the seamless integration into `R` via [`reticulate`](https://rstudio.github.io/reticulate/).

# Functionalities

## Installation and basic usage {#gettingstarted}

c-lasso is available on pip and can be installed in the shell using

```shell
pip install c_lasso
```

Below is an example of the basic usage of `c-lasso` in Python. 

```python
# Import the main class of the package
from classo import classo_problem

# Define a c-lasso problem instance with default setting, 
# given data X, y, and constraints C.
problem  = classo_problem(X,y,C)

# Add possible modifications of the problem instance 
...

# Solve the specified problem instance
problem.solve()

# Show the problem specficiation and the corresponding solution
print(problem)
print(problem.solution)
```

## Statistical problem formulations {#formulations}

Depending on the type of data and the prior assumptions on data, the noise $\epsilon$, and the model parameters, `c-lasso` allows 
different estimation problem formulations. More specifically, the package can solve the following 
four regression-type and two classification-type formulations:


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

This regression problem uses the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) $h_{\rho}$ as objective function 
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

This formulation is the default problem formulation in `c-lasso`. It is similar to [*R1*](#R1) but allows for joint estimation of the (constrained) $\beta$ vector and the standard deviation $\sigma$ in a concomitant fashion [@Combettes:2020a; @Combettes:2020b].

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

This formulation combines [*R2*](#R2) and [*R3*](#R3) allowing robust joint estimation of the (constrained) $\beta$ vector and 
the scale $\sigma$ in a concomitant fashion [@Combettes:2020a; @Combettes:2020b].

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

where $L \left((r_1,...,r_n)^T \right) := \sum_{i=1}^n l(r_i)$ and $l$ is defined as:

$$
l(r) = \begin{cases} (1-r)^2 & if \quad r \leq 1 \\ 0 &if \quad r \geq 1 \end{cases}
$$

This formulation is similar to [*R1*](#R1) but adapted for classification tasks, i.e, $y \in \{-1,1\}^n$ using the Square Hinge loss with (constrained) sparse $\beta$ vector estimation [@Lee:2013].

```python
# formulation C1
problem.formulation.huber = False
problem.formulation.concomitant = False
problem.formulation.classification = True
```

### *C2* Contrained sparse classification with Huberized Square Hinge loss: {#C2}       

$$
    \arg \min_{\beta \in \mathbb{R}^d} L_{\rho}(y^T X\beta - y) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$

where $L_{\rho} \left((r_1,...,r_n)^T \right) := \sum_{i=1}^n l_{\rho}(r_i)$ and $l_{\rho}$ is defined as :

$$
l_{\rho}(r) = \begin{cases} (1-r)^2 &if \quad \rho \leq r \leq 1 \\ (1-\rho)(1+\rho-2r) & if \quad r \leq \rho \\ 0 &if \quad r \geq 1 \end{cases}
$$

This formulation is similar to [*C1*](#C1) but uses the Huberized Square Hinge loss for robust classification with (constrained) sparse $\beta$ vector estimation [@Rosset:2007].

```python
# formulation C2
problem.formulation.huber = True
problem.formulation.concomitant = False
problem.formulation.classification = True
```

## Optimization schemes {#method}

The problem formulations *R1-C2* require different algorithmic strategies for efficiently solving the underlying optimization problems. The `c-lasso` package implements four published algorithms with provable convergence guarantees. The package also includes novel algorithmic extensions to solve Huber-type problems efficiently using the mean-shift formulation [@Mishra2019]. The following algorithmic schemes are implemented: 

- Path algorithms (*Path-Alg*): 
This algorithm follows the proposal in [@Gaines:2018;Jeon:2020]) and uses the fact that the solution path along &lambda; is piecewise-affine [@Rosset:2007]. We also provide a novel efficient procedure that allows to derive the solution for the concomitant problem *R3* along the path with little computational overhead.

- Douglas-Rachford-type splitting method (*DR*): 
This algorithm can solve all regression problems *R1-R4*. It is based on Doulgas-Rachford splitting in a higher-dimensional product space and 
makes use of the proximity operators of the perspective of the LS objective [@Combettes:2020a; @Combettes:2020b]. The Huber problem with concomitant scale *R4* is reformulated as scaled Lasso problem with mean shift vector [@Mishra:2019] and thus solved in (n + d) dimensions.

- Projected primal-dual splitting method (*P-PDS*): 
This algorithm is derived from [@Briceno:2020] and belongs to the class of proximal splitting algorithms, extending the classical Forward-Backward (FB) 
(aka proximal gradient descent) algorithm to handle an additional linear equality constraint via projection. In the absence of a linear constraint, the method reduces to FB.

- Projection-free primal-dual splitting method (*PF-PDS*):
This algorithm is a special case of an algorithm proposed in [@Combettes:2011] (Eq.4.5) and also belongs to the class of 
proximal splitting algorithms. The algorithm does not require projection operators which may be beneficial when C has a more complex structure. 
In the absence of a linear constraint, the method reduces to the Forward-Backward-Forward scheme.

The following table summarizes the available algorithms and their recommended usage for each problem: 

|             |*Path-Alg*| *DR* | *P-PDS* | *PF-PDS* |
|-|:-:|:-:|:-:|:-:|
| [*R1*](#R1) | use for large $\lambda$ and path computation | use for small $\lambda$ | possible | use for complex constraints |
| [*R2*](#R2) | use for large $\lambda$ and path computation | use for small $\lambda$ | possible | use for complex constraints |
| [*R3*](#R3) | use for large $\lambda$ and path computation | use for small $\lambda$ |    -      |        -           |
| [*R4*](#R4) |           -                                                         | only option                |    -      |  - |-
| [*C1*](#C1) | only option                                                      |                 -                |   -       |  - |
| [*C2*](#C2) | only option                                                     |                -                 |     -     |  - |



The following Python snippet shows how to select a specific algorithm: 
```python
problem.numerical_method = "Path-Alg" 
# alternative options: "DR", "P-PDS", and "PF-PDS" 
```

## Computation modes and model selection {#model}

The `c-lasso` package provides several computation modes and model selection schemes for tuning the regularization parameter.

- *Fixed Lambda*: This setting lets the user choose a fixed parameter $\lambda$ or a proportion $l \in [0,1]$ such that $\lambda = l\times \lambda_{\max}$. 
The default value is a scale-dependent tuning parameter that has been derived in [@Shi:2016] and applied in [@Combettes:2020b].

- *Path Computation*: This setting allows the computation of a solution path for $\lambda$ parameters in an interval $[\lambda_{\min}, \lambda_{\max}]$. The solution path is computed via the *Path-Alg* scheme or via warm-starts for other optimization schemes. 

[comment]: <> (This can be done much faster than by computing separately the solution for each $\lambda$ of the grid, by using the Path-alg algorithm. One can also use warm starts : starting with $\beta_0 = 0$ for $\lambda_0 = \lambda_{\max}$, and then iteratvely compute $\beta_{k+1}$ using one of the optimization schemes with $\lambda = \lambda_{k+1} := \lambda_{k} - \epsilon$ and with a warm start set to $\beta_{k}$. )

- *Cross Validation*: This setting allows the selection of the regularization parameter $\lambda$ via k-fold cross validation for $\lambda \in [\lambda_{\min}, \lambda_{\max}]$. Both the Minimum Mean Squared Error (or Deviance) (MSE)  and the "One-Standard-Error rule" (1SE) are available [@Hastie:2009].

- *Stability Selection*: This setting allows the selection of the $\lambda$ via stability selection [@Meinshausen:2010;@Lin:2014;@Combettes:2020b]. Three modes are
available for the selection of variables over subsamples: selection at a fixed $\lambda$ [@Combettes:2020b], selection of the q first variables entering the path (the default setting in `c-lasso`), and selection of the q largest coefficients (in absolute value) across the path [@Meinshausen:2010].

The python syntax to use a specific computation mode and model selection is exemplified below:

```python
# Example how to do only cross-validation and path computation:
problem.model_selection.LAMfixed = False
problem.model_selection.PATH = True
problem.model_selection.CV = True
problem.model_selection.StabSel = False

# c-lasso also allows to specify multiple model selection schemes, e.g., adding stability selection via
problem.model_selection.StabSel = True
```

Each model selection procedure has additional meta-parameters that are described in the Documentation.


# Computational examples  

## Basic workflow using synthetic data

We illustrate the workflow of the `c-lasso` package on synthetic data using the built-in routine ```random_data``` which enables the generation of test 
problem instances with normally distributed data X, sparse coefficient vectors $\beta$, and constraints $C \in R^{k\times d}$.

[comment]: <> (It generates randomly the vectors $\beta \in R^d$ , $X \in R^{n\times d}$, $C \in R^{k\times d}$ [can also be the all-one vector with the input ```zerosum``` set to true], and $y \in R^n$ normally distributed with respect to the model $C\beta=0$, $y-X\beta \sim N[0,I_n\sigma^2]$ and $\beta$ has only d_nonzero non-null componant. )

Here, we use a problem instance with $n=100$, $d=100$, a $\beta$ with five non-zero components, $\sigma=0.5$, and a zero-sum contraint. 

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

Here, we use [formulation](#formulations) [*R2*](#R2) with $\rho=1.5$, [computation mode and model selections](#model) *Fixed Lambda* with $\lambda = 0.1\lambda_{\max}$, *Path computation*, and *Stability Selection* (as per default). 

Here, the corresponding output reads: 

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

For this tuned example, the solutions at the fixed lambda and with stability selection recover the oracle solution. 
Note that the run time for this $d=100$-dimensional example for a single path computation is about 0.5 seconds on a standard Laptop.

## Log-contrast regression on gut microbiome data

We next illustrate the application of the `c-lasso` package on a microbiome dataset, considered in [@Lin:2014;@Shi:2016;@Combettes:2020b]. The task is to predict the Body Mass Index (BMI) of $n=96$ participants from $p=45$ relative abundances of bacterial genera. 


## Calling `c-lasso` in R 

The `c-lasso` package can be conveniently integrated in `R` using the `R` package ```reticulate```. A successful interfacing is already in use in the 
R package [`trac`](https://github.com/jacobbien/trac) [@Bien:2020].

The code snippet below shows how `c-lasso` is called in R to perform regression at a fixed lambda $\lambda = 0.1\lambda_{\max}$. In R, X and C should be of ```matrix``` type, and y of ```array``` type.

```r
problem<- classo$classo_problem(X=X,C=C,y=y)) 
problem$model_selection$LAMfixed <- TRUE
problem$model_selection$StabSel <- FALSE
problem$model_selection$LAMfixedparameters$rescaled_lam <- TRUE
problem$model_selection$LAMfixedparameters$lam <- 0.1
problem$solve()

# extract coefficent vector 
beta <- as.matrix(map_dfc(problem$solution$LAMfixed$beta, as.numeric))
```

# Acknowledgements

This work of LS was conducted at and financially supported by the Center for Computational Mathematics (CCM), Flatiron Institute, New York, and the Institute of Computational Biology, Helmholtz Zentrum München. We thank Dr. Leslie Greengard (CCM and Courant Institute, NYU) for facilitating the initial contact between LS and CLM. 

# References



