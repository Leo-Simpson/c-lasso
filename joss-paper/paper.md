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

Here, $X$ and $y$ are given outcome and predictor data. The vector y can be continuous (for regression) or binary (for classification). $C$ is a general constraint matrix. The vector $\beta$ comprises the unknown coefficients and $\sigma$ an unknown scale.


# Statement of need 

The package handles several estimators for inferring location and scale, including the constrained Lasso, the constrained scaled Lasso, and sparse Huber M-estimation with linear equality constraints Several algorithmic strategies, including path and proximal splitting algorithms, are implemented to solve the underlying convex optimization problems. We also include two model selection strategies for determining the sparsity of the model parameters: k-fold cross-validation and stability selection. This package is intended to fill the gap between popular python tools such as `scikit-learn` which <em>cannot</em> solve sparse constrained problems and general-purpose optimization solvers such as `cvx` that do not scale well for the considered problems or are inaccurate. We show several use cases of the package, including an application of sparse log-contrast regression tasks for compositional microbiome data. We also highlight the seamless integration of the solver into `R` via the `reticulate` package. 


# Current functionalities



## Formulations 

Depending on the prior on the solution $\beta, \sigma$ and on the noise $\epsilon$, the previous forward model can lead to different types of estimation problems. 

Our package can solve six of those : four regression-type and two classification-type formulations.




#### [R1] Standard constrained Lasso regression:             

$$
    \arg \min_{\beta \in \mathbb{R}^d} \left\lVert X\beta - y \right\rVert^2 + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$

This is the standard Lasso problem with linear equality constraints on the $\beta$ vector. 
The objective function combines Least-Squares for model fitting with l1 penalty for sparsity.   

#### [R2] Contrained sparse Huber regression:                   

$$
    \arg \min_{\beta \in \mathbb{R}^d} h_{\rho} (X\beta - y) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$

This regression problem uses the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) as objective function 
for robust model fitting with l1 and linear equality constraints on the $\beta$ vector. The parameter $\rho=1.345$.

#### [R3] Contrained scaled Lasso regression: 

$$
    \arg \min_{\beta \in \mathbb{R}^d} \frac{\left\lVert X\beta - y \right\rVert^2}{\sigma} + \frac{n}{2} \sigma + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$



This formulation is similar to [R1] but allows for joint estimation of the (constrained) $\beta$ vector and 
the standard deviation $\sigma$ in a concomitant fashion (see [References](#references) [4,5] for further info).
This is the default problem formulation in c-lasso.

#### [R4] Contrained sparse Huber regression with concomitant scale estimation:        

$$
    \arg \min_{\beta \in \mathbb{R}^d} \left( h_{\rho} \left( \frac{X\beta - y}{\sigma} \right) + n \right) \sigma + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$


This formulation combines [R2] and [R3] to allow robust joint estimation of the (constrained) $\beta$ vector and 
the scale $\sigma$ in a concomitant fashion (see [References](#references) [4,5] for further info).

#### [C1] Contrained sparse classification with Square Hinge loss: 

$$
    \arg \min_{\beta \in \mathbb{R}^d} L(y^T X\beta - y) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$

where $L \left((r_1,...,r_n)^T \right) := \sum_{i=1}^n l(r_i)$ and $l$ is defined as :

$$
l(r) = \begin{cases} (1-r)^2 & if \quad r \leq 1 \\ 0 &if \quad r \geq 1 \end{cases}
$$


This formulation is similar to [R1] but adapted for classification tasks using the Square Hinge loss with (constrained) sparse $\beta$ vector estimation.

#### [C2] Contrained sparse classification with Huberized Square Hinge loss:        


$$
    \arg \min_{\beta \in \mathbb{R}^d} L_{\rho}(y^T X\beta - y) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$

where $L_{\rho} \left((r_1,...,r_n)^T \right) := \sum_{i=1}^n l_{\rho}(r_i)$ and $l_{\rho}$ is defined as :

$$
l_{\rho}(r) = \begin{cases} (1-r)^2 &if \quad \rho \leq r \leq 1 \\ (1-\rho)(1+\rho-2r) & if \quad r \leq \rho \\ 0 &if \quad r \geq 1 \end{cases}
$$

This formulation is similar to [C1] but uses the Huberized Square Hinge loss for robust classification with (constrained) sparse $\beta$ vector estimation.



## Optimization schemes

The available problem formulations [R1-C2] require different algorithmic strategies for 
efficiently solving the underlying optimization problem. We have implemented four 
algorithms (with provable convergence guarantees) that vary in generality and are not 
necessarily applicable to all problems. For each problem type, c-lasso has a default algorithm 
setting that proved to be the fastest in our numerical experiments.

#### Path algorithms (Path-Alg) 
This is the default algorithm for non-concomitant problems [R1,R3,C1,C2]. 
The algorithm uses the fact that the solution path along &lambda; is piecewise-
affine (as shown, e.g., in [1]). When Least-Squares is used as objective function,
we derive a novel efficient procedure that allows us to also derive the 
solution for the concomitant problem [R2] along the path with little extra computational overhead.

#### Projected primal-dual splitting method (P-PDS):
This algorithm is derived from [2] and belongs to the class of 
proximal splitting algorithms. It extends the classical Forward-Backward (FB) 
(aka proximal gradient descent) algorithm to handle an additional linear equality constraint
via projection. In the absence of a linear constraint, the method reduces to FB.
This method can solve problem [R1]. For the Huber problem [R3], 
P-PDS can solve the mean-shift formulation of the problem (see [6]).

#### Projection-free primal-dual splitting method (PF-PDS):
This algorithm is a special case of an algorithm proposed in [3] (Eq.4.5) and also belongs to the class of 
proximal splitting algorithms. The algorithm does not require projection operators 
which may be beneficial when C has a more complex structure. In the absence of a linear constraint, 
the method reduces to the Forward-Backward-Forward scheme. This method can solve problem [R1]. 
For the Huber problem [R3], PF-PDS can solve the mean-shift formulation of the problem (see [6]).

#### Douglas-Rachford-type splitting method (DR)
This algorithm is the most general algorithm and can solve all regression problems 
[R1-R4]. It is based on Doulgas Rachford splitting in a higher-dimensional product space.
It makes use of the proximity operators of the perspective of the LS objective (see [4,5])
The Huber problem with concomitant scale [R4] is reformulated as scaled Lasso problem 
with the mean shift (see [6]) and thus solved in (n + d) dimensions. 


## Model selections

#### Path computation

#### Cross validation

#### Stability Selection

#### Fixed Lambda

# Basic workflow


Here is a basic example that shows how to run c-lasso on synthetic data.

#### Generate random data
The c-lasso package includes
the routine ```random_data``` that allows you to generate problem instances using normally distributed data.

```python
n,d,d_nonzero,k,sigma =100,100,5,1,0.5
(X,C,y),sol = random_data(n,d,d_nonzero,k,sigma,zerosum=True)
```
This code snippet generates a problem instance with sparse &beta; in dimension
d=100 (sparsity d_nonzero=5). The design matrix X comprises n=100 samples generated from an i.i.d standard normal
distribution. The dimension of the constraint matrix C is d x k matrix. The noise level is $\sigma$=0.5. 
The input ```zerosum=True``` implies that C is the all-ones vector and $C\beta=0$. The n-dimensional outcome vector y
and the regression vector $\beta$ is then generated to satisfy the given constraints. 

#### Define the problem
Next we can define a default c-lasso problem instance with the generated data:

```python
problem = classo_problem(X,y,C) 
```

#### Solve the optimization problem
One can solve the corresponding c-lasso problem instance using

```python
problem.solve()
```

#### Vizualise the result 
After completion, the results of the optimization and model selection routines 
can be visualized using

```python
print(problem.solution)
```

The command shows the running time(s) for the c-lasso problem instance, and the selected variables for sability selection

```
SELECTED VARIABLES : 
16
44
65
90
93
Running time : 
Running time for Path computation    : 'not computed'
Running time for Cross Validation    : 'not computed'
Running time for Stability Selection : 1.561s
Running time for Fixed LAM           : 'not computed'
```

Here, we only used stability selection as *default* model selection strategy. 
The command also allows you to inspect the computed stability profile for all variables 
at the theoretical $\lambda$

![1.StabSel](https://github.com/Leo-Simpson/Figures/blob/master/example1/StabSel.png)

and the entire $\lambda$ path (as we have used the path algorithm for optimization). We can see that stability selection
can identify the five true non-zero entries in the $\beta$ vector

![StabSel-path](https://github.com/Leo-Simpson/Figures/blob/master/example1/StabSel-path.png)


The refitted $\beta$ values on the selected support are also displayed in the next plot

![beta](https://github.com/Leo-Simpson/Figures/blob/master/example1/StabSel-beta.png)





# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.


# Acknowledgements

We acknowledge ... 

# References