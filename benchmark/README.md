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

### Douglas-Rachford-type splitting method (DR)
This algorithm is the most general algorithm and can solve all regression problems 
[R1-R4]. It is based on Doulgas Rachford splitting in a higher-dimensional product space.
It makes use of the proximity operators of the perspective of the LS objective (see [4,5])
The Huber problem with concomitant scale [R4] is reformulated as scaled Lasso problem 
with the mean shift (see [6]) and thus solved in (n + d) dimensions. 

### CVX (Conic operator splitting,scs)
For external comparison, we use cvx and its underlying conic solver (scs). For more info, see [4].



## References 

* [1] B. R. Gaines, J. Kim, and H. Zhou, [Algorithms for Fitting the Constrained Lasso](https://www.tandfonline.com/doi/abs/10.1080/10618600.2018.1473777?journalCode=ucgs20), J. Comput. Graph. Stat., vol. 27, no. 4, pp. 861–871, 2018.

* [2] L. Briceno-Arias and S.L. Rivera, [A Projected Primal–Dual Method for Solving Constrained Monotone Inclusions](https://link.springer.com/article/10.1007/s10957-018-1430-2?shared-article-renderer), J. Optim. Theory Appl., vol. 180, Issue 3, March 2019.

* [3] S. Rosset and J. Zhu, [Piecewise linear regularized solution paths](https://projecteuclid.org/euclid.aos/1185303996), Ann. Stat., vol. 35, no. 3, pp. 1012–1030, 2007.

* [4] B. O’Donoghue, E. Chu, N. Parikh, and S. Boyd. "Conic optimization via operator splitting and homogeneous self-dual embedding." Journal of Optimization Theory and Applications 169, no. 3 (2016): 1042-1068.


