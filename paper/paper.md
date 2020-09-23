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

Those are all variants of the standard formulation "*R1*" : 

$$
    \arg \min_{\beta \in \mathbb{R}^d} \left\lVert X\beta - y \right\rVert^2 + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$


- ***R1* Standard constrained Lasso regression**: 
This is the standard Lasso problem with linear equality constraints on the $\beta$ vector. 
The objective function combines Least-Squares for model fitting with $L_1$ penalty for sparsity.   

- ***R2* Contrained sparse Huber regression**: 
This regression problem uses the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) as objective function 
for robust model fitting with $L_1$ and linear equality constraints on the $\beta$ vector. The parameter $\rho$ is set to $1.345$ by default [@Aigner:1976]

- ***R3* Contrained scaled Lasso regression**: 
This formulation is similar to *R1* but allows for joint estimation of the (constrained) $\beta$ vector and 
the standard deviation $\sigma$ in a concomitant fashion [@Combettes:2020.1; @Combettes:2020.2].
This is the default problem formulation in c-lasso.

- ***R4* Contrained sparse Huber regression with concomitant scale estimation**: 
This formulation combines *R2* and *R3* to allow robust joint estimation of the (constrained) $\beta$ vector and 
the scale $\sigma$ in a concomitant fashion [@Combettes:2020.1; @Combettes:2020.2].

- ***C1* Contrained sparse classification with Square Hinge loss**: 
This formulation is similar to *R1* but adapted for classification tasks using the Square Hinge loss with (constrained) sparse $\beta$ vector estimation [@Lee:2013].

- ***C2* Contrained sparse classification with Huberized Square Hinge loss**:        
This formulation is similar to *C1* but uses the Huberized Square Hinge loss for robust classification with (constrained) sparse $\beta$ vector estimation [@Rosset:2007].

## Optimization schemes

The available problem formulations *R1-C2* require different algorithmic strategies for 
efficiently solving the underlying optimization problem. We have implemented four 
algorithms (with provable convergence guarantees) that vary in generality and are not 
necessarily applicable to all problems. For each problem type, c-lasso has a default algorithm 
setting that proved to be the fastest in our numerical experiments.

- **Path algorithms (*Path-Alg*)** : 
This is the default algorithm for non-concomitant problems *R1,R2,C1,C2*. 
The algorithm uses the fact that the solution path along &lambda; is piecewise-affine (as shown, e.g., in [@Gaines:2018]). When Least-Squares is used as objective function, we derive a novel efficient procedure that allows us to also derive the solution for the concomitant problem *R3* along the path with little extra computational overhead.

- **Projected primal-dual splitting method (*P-PDS*)** : 
This algorithm is derived from [@Briceno:2020] and belongs to the class of 
proximal splitting algorithms. It extends the classical Forward-Backward (FB) 
(aka proximal gradient descent) algorithm to handle an additional linear equality constraint
via projection. In the absence of a linear constraint, the method reduces to FB.
This method can solve problem *R1*. For the Huber problem *R2*, 
P-PDS can solve the mean-shift formulation of the problem  [@Mishra:2019].

- **Projection-free primal-dual splitting method (*PF-PDS*)** :
This algorithm is a special case of an algorithm proposed in [@Combettes:2011] (Eq.4.5) and also belongs to the class of 
proximal splitting algorithms. The algorithm does not require projection operators 
which may be beneficial when C has a more complex structure. In the absence of a linear constraint, 
the method reduces to the Forward-Backward-Forward scheme. This method can solve problem *R1*. 
For the Huber problem *R2*, PF-PDS can solve the mean-shift formulation of the problem [@Mishra:2019].

- **Douglas-Rachford-type splitting method (*DR*)** : 
This algorithm is the most general algorithm and can solve all regression problems 
*R1-R4*. It is based on Doulgas Rachford splitting in a higher-dimensional product space.
It makes use of the proximity operators of the perspective of the LS objective [@Combettes:2020.1; @Combettes:2020.2].
The Huber problem with concomitant scale *R4* is reformulated as scaled Lasso problem 
with the mean shift [@Mishra:2019] and thus solved in (n + d) dimensions. 


## Model selections

Different models are implemented together with the optimization schemes, to overcome the difficulty of choosing the penalization free parameter $\lambda$. 

- *Fixed Lambda* : This approach is simply letting the user choose the parameter $\lambda$, or to choose $l \in [0,1]$ such that $\lambda = l\times \lambda_{\max}$. 
The default value is a scale-dependent tuning parameter that has been proposed in [Combettes:2020.2] and derived in [@Shi:2016].

- *Path Computation* :The package also leaves the possibility to us to compute the solution for a range of $\lambda$ parameters in an interval $[\lambda_{\min}, \lambda_{\max}]$. It can be done using *Path-Alg* or warm-start with any other optimization scheme. 

[comment]: <> (This can be done much faster than by computing separately the solution for each $\lambda$ of the grid, by using the Path-alg algorithm. One can also use warm starts : starting with $\beta_0 = 0$ for $\lambda_0 = \lambda_{\max}$, and then iteratvely compute $\beta_{k+1}$ using one of the optimization schemes with $\lambda = \lambda_{k+1} := \lambda_{k} - \epsilon$ and with a warm start set to $\beta_{k}$. )

- *Cross Validation* : Then one can use a model selection, to choose the appropriate penalisation. This can be done by using k-fold cross validation to find the best $\lambda \in [\lambda_{\min}, \lambda_{\max}]$ with or without "one-standard-error rule" [@Hastie:2009].

- *Stability Selection* : Another variable selection model than can be used is stability selection [@Lin:2014; @Meinshausen:2010; Combettes:2020.2].

# Basic workflow


Here is a basic example that shows how to run c-lasso on synthetic data.

c-lasso is available on pip, one can install it using ```pip install c_lasso```. Then on python, to import the package, one should use ```import classo```

Let us now begin the tutorial. 
Firstly, let us generate a dataset using
the routine ```random_data``` included in the c-lasso package, that allows you to generate instances using normally distributed data.

```python
>>> n,d,d_nonzero,k,sigma =100,100,5,1,0.5
>>> (X,C,y),sol = classo.random_data(n,d,d_nonzero,k,sigma,zerosum=True, seed = 123 )
>>> list(numpy.nonzero(sol))
[43, 47, 74, 79, 84]
```
This code snippet generates the vectors $\beta \in R^d$ , $X \in R^{n\times d}$ , $C \in R^{k\times d}$ (here it is actually all-one vector because of the input ```zerosum```), and $y \in R^n$ normally distributed with respect to the model $C\beta=0$, $y-X\beta \sim N(0,\sigma)$ and $\beta$ has only d_nonzero non-null componant (which are plot exmplicetly above).


Then, let us define a ```classo_problem``` instance with the generated dataset in order to formulate the optimization problem we want to solve. 

```python
>>> problem  = classo.classo_problem(X,y,C)  # define a c-lasso problem instance with default setting

>>> problem.formulation.huber  = True
>>> problem.formulation.concomitant = False
>>> problem.formulation.rho = 1.5  # change the formulation of the problem

>>> problem.model_selection.LAMfixed = True # add the computation for a fixed lambda
>>> problem.model_selection.LAMfixedparameters.rescaled_lam = True
>>> problem.model_selection.LAMfixedparameters.lam = 0.1 # lambda is 0.1*lambdamax

>>> problem.model_selection.PATH = True #add the computation of the lambda-path

>>> problem.solve() # solve our optimization problem
```


Here, we have modified the [formulation](##formulations) of the problem in order to use *R2*, with $\rho=1.5$. 
We have chosen the following [model selections](##model-selections) : *Fixed Lambda* with $\lambda = 0.1\lambda_{\max}$ ; *Path computation* and *Stability Selection* which is computed by default. 
Then, those problems are solved using the method ```solve()``` which computes everything. 

Finally, one can visualize the solutions and see the running time, and the name of the selected variables by calling the instance ```problem.solution```. Note that by calling directly the instance ```problem``` one could also visualize the main parameters of the optimization problems one is solving. In our case, the running time is in the order of 0.1sec for the fixed lambda and path computation, but vary from 2sec to 4sec for the stability selection computation.  


![Graphics plotted after calling ```problem.solution``` ](figures/figure-concat.png)

Let us that the models have recovered the right variables and the computation have been done quickly, which is comforting, but not surprising because in this example the noise is little and the number of variable is still small. 


# Acknowledgements

We acknowledge ... 

# References