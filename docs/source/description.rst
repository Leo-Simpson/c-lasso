
Mathematical description
=============================


The forward model is assumed to be: 

.. math::
   y = X \beta + \sigma \epsilon \qquad \textrm{subject to} \qquad C\beta=0

Here, y and X are given outcome and predictor data. The vector y can be continuous (for regression) or binary (for classification). C is a general constraint matrix. The vector :math:`\beta` comprises the unknown coefficients and :math:`\sigma` an 
unknown scale.

The package handles several different estimators for inferring :math:`\beta` and :math:`\sigma`), including 
the constrained Lasso, the constrained scaled Lasso, and sparse Huber M-estimation with linear equality constraints.
Several different algorithmic strategies, including path and proximal splitting algorithms, are implemented to solve 
the underlying convex optimization problems.

We also include two model selection strategies for determining the sparsity of the model parameters: k-fold cross-validation and stability selection.   

This package is intended to fill the gap between popular python tools such as `scikit-learn <https://scikit-learn.org/stable/>`_ which CANNOT solve sparse constrained problems and general-purpose optimization solvers that do not scale well for the considered problems.

Below we show several use cases of the package, including an application of sparse *log-contrast*
regression tasks for *compositional* microbiome data.

The code builds on results from several papers which can be found in the [References](#references). We also refer to the accompanying `JOSS paper submission <https://github.com/Leo-Simpson/c-lasso/blob/master/paper/paper.md>`_, also available on `arXiv <https://arxiv.org/pdf/2011.00898.pdf>`_.


Regression and classification problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The c-lasso package can solve six different types of estimation problems: 
four regression-type and two classification-type formulations.

[R1] Standard constrained Lasso regression
"""""""""""""""""""""""""""""""""""""""""""""          

.. math::
   \min_{\beta \in \mathbb{R}^d} \left\lVert X\beta - y \right\rVert^2 + \lambda \left\lVert \beta\right\rVert_1 \qquad \textrm{subject to} \qquad  C\beta = 0


This is the standard Lasso problem with linear equality constraints on the :math:`\beta` vector. 
The objective function combines Least-Squares for model fitting with l1 penalty for sparsity.   

[R2] Constrained sparse Huber regression
""""""""""""""""""""""""""""""""""""""""""""""""""                   

.. math::
   \min_{\beta \in \mathbb{R}^d} h_{\rho} (X\beta - y) + \lambda \left\lVert \beta\right\rVert_1 \qquad \textrm{subject to} \qquad  C\beta = 0

This regression problem uses the `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_ as objective function 
for robust model fitting with l1 and linear equality constraints on the :math:`\beta` vector. The parameter :math:`\rho=1.345`.

[R3] Constrained scaled Lasso regression
""""""""""""""""""""""""""""""""""""""""""""""""""  

.. math::
   \min_{\beta \in \mathbb{R}^d, \sigma \in \mathbb{R}_{0}} \frac{\left\lVert X\beta - y \right\rVert^2}{\sigma} + \frac{n}{2} \sigma + \lambda \left\lVert \beta\right\rVert_1 \qquad \textrm{subject to} \qquad  C\beta = 0


This formulation is similar to [R1] but allows for joint estimation of the (constrained) :math:`\beta` vector and the standard deviation :math:`\sigma` in a concomitant fashion [4]_, [5]_ .
This is the default problem formulation in c-lasso.

[R4] Constrained sparse Huber regression with concomitant scale estimation 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""       

.. math::
   \min_{\beta \in \mathbb{R}^d, \sigma \in  \mathbb{R}_{0}} \left( h_{\rho} \left( \frac{X\beta - y}{\sigma} \right) + n \right) \sigma + \lambda \left\lVert \beta\right\rVert_1 \qquad \textrm{subject to} \qquad  C\beta = 0

This formulation combines [R2] and [R3] to allow robust joint estimation of the (constrained) :math:`\beta` vector and 
the scale :math:`\sigma` in a concomitant fashion [4]_ , [5]_ .

[C1] Constrained sparse classification with Square Hinge loss
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""  

.. math::
   \min_{\beta \in \mathbb{R}^d} \sum_{i=1}^n l(y_i x_i^\top\beta) + \lambda \left\lVert \beta\right\rVert_1 \qquad \textrm{subject to} \qquad  C\beta = 0


where the :math:`x_i` are the rows of X and l is defined as:

.. math::
   l(r) = \begin{cases} (1-r)^2 & if \quad r \leq 1 \\ 0 &if \quad r \geq 1 \end{cases}

This formulation is similar to [R1] but adapted for classification tasks using the Square Hinge loss
with (constrained) sparse :math:`\beta` vector estimation.

[C2] Constrained sparse classification with Huberized Square Hinge loss
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""  

.. math::
   \min_{\beta \in \mathbb{R}^d}  \sum_{i=1}^n  l_{\rho}(y_i x_i^\top\beta) + \lambda \left\lVert \beta\right\rVert_1 \qquad \textrm{subject to} \qquad  C\beta = 0 \,.

where the :math:`x_i`  are the rows of X and :math:`l_{\rho}` is defined as:

.. math::
   l_{\rho}(r) = \begin{cases} (1-r)^2 &if \quad \rho \leq r \leq 1 \\ (1-\rho)(1+\rho-2r) & if \quad r \leq \rho \\ 0 &if \quad r \geq 1 \end{cases}

This formulation is similar to [C1] but uses the Huberized Square Hinge loss for robust classification with (constrained) sparse :math:`\beta` vector estimation [7]_.



Optimization schemes
^^^^^^^^^^^^^^^^^^^^^^^^^

The available problem formulations [R1-C2] require different algorithmic strategies for 
efficiently solving the underlying optimization problem. We have implemented four 
algorithms (with provable convergence guarantees) that vary in generality and are not 
necessarily applicable to all problems. For each problem type, c-lasso has a default algorithm 
setting that proved to be the fastest in our numerical experiments.

Path algorithms (Path-Alg) 
""""""""""""""""""""""""""""""""""""""""""""""""""  
This is the default algorithm for non-concomitant problems [R1,R3,C1,C2]. 
The algorithm uses the fact that the solution path along :math:`\lambda` is piecewise-
affine [1]_. When Least-Squares is used as objective function,
we derive a novel efficient procedure that allows us to also derive the 
solution for the concomitant problem [R2] along the path with little extra computational overhead.

Projected primal-dual splitting method (P-PDS)
""""""""""""""""""""""""""""""""""""""""""""""""""  
This algorithm is derived from [2]_ and belongs to the class of 
proximal splitting algorithms. It extends the classical Forward-Backward (FB) 
(aka proximal gradient descent) algorithm to handle an additional linear equality constraint
via projection. In the absence of a linear constraint, the method reduces to FB.
This method can solve problem [R1]. For the Huber problem [R3], 
P-PDS can solve the mean-shift formulation of the problem [6]_.

Projection-free primal-dual splitting method (PF-PDS)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This algorithm is a special case of an algorithm proposed in [3]_ (Eq.4.5) and also belongs to the class of 
proximal splitting algorithms. The algorithm does not require projection operators 
which may be beneficial when C has a more complex structure. In the absence of a linear constraint, 
the method reduces to the Forward-Backward-Forward scheme. This method can solve problem [R1]. 
For the Huber problem [R3], PF-PDS can solve the mean-shift formulation of the problem [6]_.

Douglas-Rachford-type splitting method (DR)
""""""""""""""""""""""""""""""""""""""""""""""""""  
This algorithm is the most general algorithm and can solve all regression problems 
[R1-R4]. It is based on Doulgas Rachford splitting in a higher-dimensional product space.
It makes use of the proximity operators of the perspective of the LS objective (see [4]_ and [5]_)
The Huber problem with concomitant scale [R4] is reformulated as scaled Lasso problem 
with the mean shift [6]_ and thus solved in (n + d) dimensions. 




References
^^^^^^^^^^^

.. [1] B. R. Gaines, J. Kim, and H. Zhou, `Algorithms for Fitting the Constrained Lasso <https://www.tandfonline.com/doi/abs/10.1080/10618600.2018.1473777?journalCode=ucgs20>`_, J. Comput. Graph. Stat., vol. 27, no. 4, pp. 861–871, 2018.
.. [2] L. Briceno-Arias and S.L. Rivera, `A Projected Primal–Dual Method for Solving Constrained Monotone Inclusions <https://link.springer.com/article/10.1007/s10957-018-1430-2?shared-article-renderer>`_, J. Optim. Theory Appl., vol. 180, Issue 3, March 2019.
.. [3] P. L. Combettes and J.C. Pesquet, `Primal-Dual Splitting Algorithm for Solving Inclusions with Mixtures of Composite, Lipschitzian, and Parallel-Sum Type Monotone Operators <https://arxiv.org/pdf/1107.0081.pdf>`_, Set-Valued and Variational Analysis, vol. 20, pp. 307-330, 2012.
.. [4] P. L. Combettes and C. L. Müller, `Perspective M-estimation via proximal decomposition <https://arxiv.org/abs/1805.06098>`_, Electronic Journal of Statistics, 2020, `Journal version <https://projecteuclid.org/euclid.ejs/1578452535>`_ 
.. [5] P. L. Combettes and C. L. Müller, `Regression models for compositional data: General log-contrast formulations, proximal optimization, and microbiome data applications <https://arxiv.org/abs/1903.01050>`_, Statistics in Bioscience, 2020.
.. [6] A. Mishra and C. L. Müller, `Robust regression with compositional covariates <https://arxiv.org/abs/1909.04990>`_, arXiv, 2019.
.. [7] S. Rosset and J. Zhu, `Piecewise linear regularized solution paths <https://projecteuclid.org/euclid.aos/1185303996>`_, Ann. Stat., vol. 35, no. 3, pp. 1012–1030, 2007.
