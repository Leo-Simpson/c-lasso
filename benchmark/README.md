<img src="https://i.imgur.com/2nGwlux.png" alt="c-lasso" height="150" align="right"/>

# Numerical benchmarks for c-lasso 

We provide numerical benchmarks for the [c-lasso package](https://c-lasso.readthedocs.io/en/latest/) in comparison to [cvxpy](https://www.cvxpy.org). 
We report run times, achieved minimum function values (with the path algorithm solutin as baseline), and constraint satisfaction quality of the zero-sum constraint.  

## Table of Contents

* [Benchmark setup](#installation)
* [Results](#getting-started)
* [Optimization schemes](#optimization-schemes)

* [References](#references)


##  Benchmark set-up

###  Tested Regression and classification problems

Here, we consider the following problem formulations (see [here](../README.md) for detailed infos):

#### [R1] Standard constrained Lasso regression:             

<img src="https://latex.codecogs.com/gif.latex?\arg\min_{\beta\in&space;R^d}&space;||&space;X\beta-y&space;||^2&space;&plus;&space;\lambda&space;||\beta||_1&space;\qquad\mbox{s.t.}\qquad&space;C\beta=0" />

#### [R2] Contrained sparse Huber regression:                   

<img src="https://latex.codecogs.com/gif.latex?\arg\min_{\beta\in&space;R^d}&space;h_{\rho}(X\beta-y&space;)&space;&plus;&space;\lambda&space;||\beta||_1&space;\qquad\mbox{s.t.}\qquad&space;C\beta=0" />

#### [C1] Contrained sparse classification with Square Hinge loss (L1-Regularized Square-Hinge SVM): 

<img src="https://latex.codecogs.com/gif.latex?\arg&space;\min_{\beta&space;\in&space;\mathbb{R}^d}&space;\sum_{i=1}^n&space;l(y_i&space;x_i^\top&space;\beta)&space;&plus;&space;\lambda&space;\left\lVert&space;\beta\right\rVert_1&space;\qquad&space;s.t.&space;\qquad&space;C\beta&space;=&space;0" title="\arg \min_{\beta \in \mathbb{R}^d} \sum_{i=1}^n l(y_i x_i \beta) + \lambda \left\lVert \beta\right\rVert_1 \qquad s.t. \qquad C\beta = 0" />

where the x<sub>i</sub> are the rows of X and l is defined as:

<img src="https://latex.codecogs.com/gif.latex?l(r)&space;=&space;\begin{cases}&space;(1-r)^2&space;&&space;if&space;\quad&space;r&space;\leq&space;1&space;\\&space;0&space;&if&space;\quad&space;r&space;\geq&space;1&space;\end{cases}" title="l(r) = \begin{cases} (1-r)^2 & if \quad r \leq 1 \\ 0 &if \quad r \geq 1 \end{cases}" />

###  Synthetic data generation and test problem set-ups 

We use the `random_data` function in c-lasso to generate X and y. We use the standard `zeroSum` constraint. We vary the number of samples n and dimensionionality d of the problems. The regularization parameter is fixed to &lambda;=0.1. This setting does not favor the path algorithm. The reported performance is thus rather a lower bound on the actual speed-up. Since, for most model selection schemes, the computation of the entire solution path is required, the path algorithm formulation would be even more preferable then.  

## Results

The running times of the micro-benchmark has been computed using Python 3.9.1 on a laptop `MacBook Air`, operating on macOS high Sierra with the processor `1,8 GHz Intel Core i5`, with memory of `8 Go 1600 MHz DDR3`.

#### R1

Run times for R1. 

![Run times on R1](./output/bm-R1-times.png)

Achieved minimum function values on R1. We observe considerable inaccuracies in cvx solutions.

![Achieved function values on C1](./output/bm-R1-losses.png)

Satistifaction of the zero-sum constraint.

![Satistifaction of the zero-sum constraint](./output/bm-R1-constraint.png)


#### R2

Run times for R2. 

![Run times on R2](./output/bm-R2-times.png)

Achieved minimum function values on R2. 

![Achieved function values on R2](./output/bm-R2-losses.png)

Satistifaction of the zero-sum constraint
![Satistifaction of the zero-sum constraint](./output/bm-R2-constraint.png)

#### C1

Run times for C1. 

![Run times on C1](./output/bm-C1-times.png)

Achieved minimum function values on C1. 

![Achieved function values on C1](./output/bm-C1-losses.png)

Satistifaction of the zero-sum constraint
![Satistifaction of the zero-sum constraint](./output/bm-C1-constraint.png)



## Optimization schemes

We consider the following schemes in the benchmark.

### Path algorithms (Path-Alg, pa) 
This is the default algorithm for non-concomitant problems [R1,R3,C1,C2]. 
The algorithm uses the fact that the solution path along &lambda; is piecewise-
affine (as shown, e.g., in [1]). When Least-Squares is used as objective function,
we derive a novel efficient procedure that allows us to also derive the 
solution for the concomitant problem [R2] along the path with little extra computational overhead.

### Projected primal-dual splitting method (pds):
This algorithm is derived from [2] and belongs to the class of 
proximal splitting algorithms. It extends the classical Forward-Backward (FB) 
(aka proximal gradient descent) algorithm to handle an additional linear equality constraint
via projection. In the absence of a linear constraint, the method reduces to FB.
This method can solve problem [R1]. 

### Douglas-Rachford-type splitting method (dr)
This algorithm is the most general algorithm and can solve all regression problems 
[R1-R4]. It is based on Doulgas Rachford splitting in a higher-dimensional product space.
It makes use of the proximity operators of the perspective of the LS objective (see [4,5])


### Operator splitting conic solver (SCS) in CVX (cvx)
For external comparison, we use cvx and its underlying conic solver (scs). For more info, see [6].



## References 

* [1] B. R. Gaines, J. Kim, and H. Zhou, [Algorithms for Fitting the Constrained Lasso](https://www.tandfonline.com/doi/abs/10.1080/10618600.2018.1473777?journalCode=ucgs20), J. Comput. Graph. Stat., vol. 27, no. 4, pp. 861–871, 2018.

* [2] L. Briceno-Arias and S.L. Rivera, [A Projected Primal–Dual Method for Solving Constrained Monotone Inclusions](https://link.springer.com/article/10.1007/s10957-018-1430-2?shared-article-renderer), J. Optim. Theory Appl., vol. 180, Issue 3, March 2019.

* [3] S. Rosset and J. Zhu, [Piecewise linear regularized solution paths](https://projecteuclid.org/euclid.aos/1185303996), Ann. Stat., vol. 35, no. 3, pp. 1012–1030, 2007.

* [4] P. L. Combettes and C. L. Müller, [Perspective M-estimation via proximal decomposition](https://arxiv.org/abs/1805.06098), Electronic Journal of Statistics, 2020, [Journal version](https://projecteuclid.org/euclid.ejs/1578452535) 

* [5] P. L. Combettes and C. L. Müller, [Regression models for compositional data: General log-contrast formulations, proximal optimization, and microbiome data applications](https://arxiv.org/abs/1903.01050), Statistics in Bioscience, 2020.

* [6] B. O’Donoghue, E. Chu, N. Parikh, and S. Boyd. [Conic optimization via operator splitting and homogeneous self-dual embedding.](https://link.springer.com/article/10.1007/s10957-016-0892-3), Journal of Optimization Theory and Applications 169, no. 3 (2016): 1042-1068.


