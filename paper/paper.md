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
c-lasso is a Python package that enables sparse and robust linear
regression and classification with linear equality constraints. 


The forward model is assumed to be:

$$
y = X \beta + \sigma \epsilon \qquad \textrm{s.t.} \qquad C\beta=0
$$

Here, $X$ and $y$ are given outcome and predictor data. The vector $y$ can be continuous (for regression) or binary (for classification). $C$ is a general constraint matrix. The vector $\beta$ comprises the unknown coefficients $\epsilon$ an unknown noise and $\sigma$ an unknown scale.

Depending on the prior we assume on those unknown variables, this forward model can lead to different types of estimation. Our package can solve six of those : four regression-type and two classification-type formulations. Those are all variants of the standard formulation "*R1*" : 

$$
    \arg \min_{\beta \in \mathbb{R}^d} \left\lVert X\beta - y \right\rVert^2 + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad  C\beta = 0
$$


### Formulations
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


### Model selections
Different models are implemented together with the optimization schemes, to overcome the difficulty of choosing the penalization free parameter $\lambda$. 

- *Fixed Lambda* : This approach is simply letting the user choose the parameter $\lambda$, or to choose $l \in [0,1]$ such that $\lambda = l\times \lambda_{\max}$. 
The default value is a scale-dependent tuning parameter that has been proposed in [Combettes:2020.2] and derived in [@Shi:2016].

- *Path Computation* :The package also leaves the possibility to us to compute the solution for a range of $\lambda$ parameters in an interval $[\lambda_{\min}, \lambda_{\max}]$. It can be done using *Path-Alg* or warm-start with any other optimization scheme. 

[comment]: <> (This can be done much faster than by computing separately the solution for each $\lambda$ of the grid, by using the Path-alg algorithm. One can also use warm starts : starting with $\beta_0 = 0$ for $\lambda_0 = \lambda_{\max}$, and then iteratvely compute $\beta_{k+1}$ using one of the optimization schemes with $\lambda = \lambda_{k+1} := \lambda_{k} - \epsilon$ and with a warm start set to $\beta_{k}$. )

- *Cross Validation* : Then one can use a model selection, to choose the appropriate penalisation. This can be done by using k-fold cross validation to find the best $\lambda \in [\lambda_{\min}, \lambda_{\max}]$ with or without "one-standard-error rule" [@Hastie:2009].

- *Stability Selection* : Another variable selection model than can be used is stability selection [@Lin:2014; @Meinshausen:2010; Combettes:2020.2].



# Statement of need 
The package handles several estimators for inferring location and scale, including the constrained Lasso, the constrained scaled Lasso, and sparse Huber M-estimation with linear equality constraints Several algorithmic strategies, including path and proximal splitting algorithms, are implemented to solve the underlying convex optimization problems. We also include two model selection strategies for determining the sparsity of the model parameters: k-fold cross-validation and stability selection. This package is intended to fill the gap between popular python tools such as `scikit-learn` which <em>cannot</em> solve sparse constrained problems and general-purpose optimization solvers such as `cvx` that do not scale well for the considered problems or are inaccurate. We show several use cases of the package, including an application of sparse log-contrast regression tasks for compositional microbiome data. We also highlight the seamless integration of the solver into `R` via the `reticulate` package. 




# Basic workflow
Here is a basic example that shows how to run c-lasso on synthetic data.

c-lasso is available on pip, one can install it using ```pip install c_lasso```. Then on python, to import the package, one should use ```import classo```

Let us now begin the tutorial. 
Firstly, let us generate a dataset using
the routine ```random_data``` included in the c-lasso package, that allows you to generate instances using normally distributed data.

```python
>>> n,d,d_nonzero,k,sigma =100,100,5,1,0.5
>>> (X,C,y),sol = random_data(n,d,d_nonzero,k,sigma,zerosum=True, seed = 123 )
>>> list(numpy.nonzero(sol))
[43, 47, 74, 79, 84]
```
This code snippet generates randomly the vectors $\beta \in R^d$ , $X \in R^{n\times d}$ , $C \in R^{k\times d}$ (here it is the all-one vector instead because of the input ```zerosum```), and $y \in R^n$ normally distributed with respect to the model $C\beta=0$, $y-X\beta \sim N(0,I_n\sigma^2)$ and $\beta$ has only d_nonzero non-null componant (which are plot above).


Then, let us define a ```classo_problem``` instance with the generated dataset in order to formulate the optimization problem we want to solve. 

```python
# to define a c-lasso problem instance with default setting : 
>>> problem  = classo_problem(X,y,C)  
# to change the formulation of the problem :
>>> problem.formulation.huber  = True
>>> problem.formulation.concomitant = False
>>> problem.formulation.rho = 1.5  
# to add the computation for a fixed lambda :
>>> problem.model_selection.LAMfixed = True 
# to set lambda to 0.1*lambdamax : 
>>> problem.model_selection.LAMfixedparameters.rescaled_lam = True
>>> problem.model_selection.LAMfixedparameters.lam = 0.1 
# to add the computation of the lambda-path : 
>>> problem.model_selection.PATH = True 
# to solve our optimization problem : 
>>> problem.solve() 
```


Here, we have modified the [formulation](###formulations) of the problem in order to use *R2*, with $\rho=1.5$. 
We have chosen the following [model selections](###model-selections) : *Fixed Lambda* with $\lambda = 0.1\lambda_{\max}$ ; *Path computation* and *Stability Selection* which is computed by default. 
Then, those problems are solved using the recommanded optimization scheme on each model according to the formulation and the size of the parameter $\lambda$

Finally, one can visualize the solutions and see the running time, and the name of the selected variables by calling the instance ```problem.solution```. Note that by calling directly the instance ```problem``` one could also visualize the main parameters of the optimization problems one is solving. In our case, the running time is in the order of 0.1sec for the fixed lambda and path computation, but vary from 2sec to 4sec for the stability selection computation.  


![Graphics plotted after calling ```problem.solution``` ](figures/figure-concat.png)

Let us remark that the models have recovered the right variables and the computation have been done quickly, which is comforting, but not surprising because in this example the noise is little and the number of variable is still small. 


# References