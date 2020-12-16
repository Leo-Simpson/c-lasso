<img src="https://i.imgur.com/2nGwlux.png" alt="c-lasso" height="150" align="right"/>

# Numerical benchmarks for c-lasso 


Here, we summarize numerical benchmarks for the [c-lasso package](https://c-lasso.readthedocs.io/en/latest/) in comparison to cvx.

## Table of Contents

* [Benchmark setup](#installation)
* [Regression and classification problems](#regression-and-classification-problems)
* [Results](#getting-started)
* [Optimization schemes](#optimization-schemes)

* [References](#references)


##  Benchmark set-up

###  Tested Regression and classification problems

#### [R1] Standard constrained Lasso regression:             

<img src="https://latex.codecogs.com/gif.latex?\arg\min_{\beta\in&space;R^d}&space;||&space;X\beta-y&space;||^2&space;&plus;&space;\lambda&space;||\beta||_1&space;\qquad\mbox{s.t.}\qquad&space;C\beta=0" />

This is the standard Lasso problem with linear equality constraints on the &beta; vector. 
The objective function combines Least-Squares for model fitting with l1 penalty for sparsity.   

#### [R2] Contrained sparse Huber regression:                   

<img src="https://latex.codecogs.com/gif.latex?\arg\min_{\beta\in&space;R^d}&space;h_{\rho}(X\beta-y&space;)&space;&plus;&space;\lambda&space;||\beta||_1&space;\qquad\mbox{s.t.}\qquad&space;C\beta=0" />

This regression problem uses the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) as objective function 
for robust model fitting with l1 and linear equality constraints on the &beta; vector. The parameter &rho;=1.345.

#### [C1] Contrained sparse classification with Square Hinge loss (Square-Hinge SVM): 

<img src="https://latex.codecogs.com/gif.latex?\arg&space;\min_{\beta&space;\in&space;\mathbb{R}^d}&space;\sum_{i=1}^n&space;l(y_i&space;x_i^\top&space;\beta)&space;&plus;&space;\lambda&space;\left\lVert&space;\beta\right\rVert_1&space;\qquad&space;s.t.&space;\qquad&space;C\beta&space;=&space;0" title="\arg \min_{\beta \in \mathbb{R}^d} \sum_{i=1}^n l(y_i x_i \beta) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad C\beta = 0" />

where the x<sub>i</sub> are the rows of X and l is defined as:

<img src="https://latex.codecogs.com/gif.latex?l(r)&space;=&space;\begin{cases}&space;(1-r)^2&space;&&space;if&space;\quad&space;r&space;\leq&space;1&space;\\&space;0&space;&if&space;\quad&space;r&space;\geq&space;1&space;\end{cases}" title="l(r) = \begin{cases} (1-r)^2 & if \quad r \leq 1 \\ 0 &if \quad r \geq 1 \end{cases}" />

This formulation is similar to [R1] but adapted for classification tasks using the Square Hinge loss
with (constrained) sparse &beta; vector estimation.

#### [C2] Contrained sparse classification with Huberized Square Hinge loss (Huberized Square-Hinge SVM):        

<img src="https://latex.codecogs.com/gif.latex?\arg&space;\min_{\beta&space;\in&space;\mathbb{R}^d}&space;\sum_{i=1}^n&space;l_{\rho}(y_i&space;x_i^\top\beta)&space;&plus;&space;\lambda&space;\left\lVert&space;\beta\right\rVert_1&space;\qquad&space;s.t.&space;\qquad&space;C\beta&space;=&space;0" title="\arg \min_{\beta \in \mathbb{R}^d} \sum_{i=1}^n l_{\rho}(y_i x_i\beta) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad C\beta = 0" />

where the x<sub>i</sub> are the rows of X and l<sub>ρ</sub> is defined as:

<img src="https://latex.codecogs.com/gif.latex?l_{\rho}(r)&space;=&space;\begin{cases}&space;(1-r)^2&space;&if&space;\quad&space;\rho&space;\leq&space;r&space;\leq&space;1&space;\\&space;(1-\rho)(1&plus;\rho-2r)&space;&&space;if&space;\quad&space;r&space;\leq&space;\rho&space;\\&space;0&space;&if&space;\quad&space;r&space;\geq&space;1&space;\end{cases}" title="l_{\rho}(r) = \begin{cases} (1-r)^2 &if \quad \rho \leq r \leq 1 \\ (1-\rho)(1+\rho-2r) & if \quad r \leq \rho \\ 0 &if \quad r \geq 1 \end{cases}" />


This formulation is similar to [C1] but uses the Huberized Square Hinge loss for robust classification 
with (constrained) sparse &beta; vector estimation.


## Results

#### R1


![Run times on R1](./output/bm-R1-times.png)

![Run times on R2](./output/bm-R2-times.png)

![Run times on C1]](./output/bm-C1-times.png)




<!---
<img src="https://i.imgur.com/8tFmM8T.png" alt="Central Park Soil Microbiome" height="250" align="right"/>
#### pH prediction using the Central Park soil dataset 
The next microbiome example considers the [Central Park Soil dataset](./examples/pH_data) from [Ramirez et al.](https://royalsocietypublishing.org/doi/full/10.1098/rspb.2014.1988). The sample locations are shown in the Figure on the right.)
-->

#### pH prediction using the 88 soils dataset

The next microbiome example considers the [88 soils dataset](./examples/pH_data) from [Lauber et al., 2009](https://pubmed.ncbi.nlm.nih.gov/19502440/).

The task is to predict pH concentration in the soil from microbial abundance data. A similar analysis is available
in [Tree-Aggregated Predictive Modeling of Microbiome Data](https://www.biorxiv.org/content/10.1101/2020.09.01.277632v1) 
with Central Park soil data from [Ramirez et al.](https://royalsocietypublishing.org/doi/full/10.1098/rspb.2014.1988).

Code to run this application is available in [the accompanying notebook](./examples/example-notebook.ipynb) under `pH data`. Below is a summary of a c-lasso problem instance (using the R3 formulation).
 
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

The c-lasso estimation results are summarized below:

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

![Ex4.1](https://github.com/Leo-Simpson/c-lasso/blob/master/figures/examplePH/R3-Beta-path.png)

![Ex4.2](https://github.com/Leo-Simpson/c-lasso/blob/master/figures/examplePH/R3-Sigma-path.png)

![Ex4.3](https://github.com/Leo-Simpson/c-lasso/blob/master/figures/examplePH/R3-StabSel.png)

![Ex4.4](https://github.com/Leo-Simpson/c-lasso/blob/master/figures/examplePH/R3-StabSel-beta.png)

![Ex4.5](https://github.com/Leo-Simpson/c-lasso/blob/master/figures/examplePH/R3-beta.png)


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



## References 

* [1] B. R. Gaines, J. Kim, and H. Zhou, [Algorithms for Fitting the Constrained Lasso](https://www.tandfonline.com/doi/abs/10.1080/10618600.2018.1473777?journalCode=ucgs20), J. Comput. Graph. Stat., vol. 27, no. 4, pp. 861–871, 2018.

* [2] L. Briceno-Arias and S.L. Rivera, [A Projected Primal–Dual Method for Solving Constrained Monotone Inclusions](https://link.springer.com/article/10.1007/s10957-018-1430-2?shared-article-renderer), J. Optim. Theory Appl., vol. 180, Issue 3, March 2019.

* [3] P. L. Combettes and J.C. Pesquet, [Primal-Dual Splitting Algorithm for Solving Inclusions with Mixtures of Composite, Lipschitzian, and Parallel-Sum Type Monotone Operators](https://arxiv.org/pdf/1107.0081.pdf), Set-Valued and Variational Analysis, vol. 20, pp. 307-330, 2012.

* [4] P. L. Combettes and C. L. Müller, [Perspective M-estimation via proximal decomposition](https://arxiv.org/abs/1805.06098), Electronic Journal of Statistics, 2020, [Journal version](https://projecteuclid.org/euclid.ejs/1578452535) 

* [5] P. L. Combettes and C. L. Müller, [Regression models for compositional data: General log-contrast formulations, proximal optimization, and microbiome data applications](https://arxiv.org/abs/1903.01050), Statistics in Bioscience, 2020.

* [6] A. Mishra and C. L. Müller, [Robust regression with compositional covariates](https://arxiv.org/abs/1909.04990), arXiv, 2019.

* [7] S. Rosset and J. Zhu, [Piecewise linear regularized solution paths](https://projecteuclid.org/euclid.aos/1185303996), Ann. Stat., vol. 35, no. 3, pp. 1012–1030, 2007.

* [8] J. Bien, X. Yan, L. Simpson, and C. L. Müller,   [Tree-Aggregated Predictive Modeling of Microbiome Data](https://www.biorxiv.org/content/10.1101/2020.09.01.277632v1), biorxiv, 2020.


