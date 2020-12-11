.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_advanced_example.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_advanced_example.py:


Advanced example
==================

Let's present how one can specify different aspects of the problem 
formulation and model selection strategy on classo, using synthetic data.


.. code-block:: default


    from classo import classo_problem, random_data
    import numpy as np








Generate the data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This code snippet generates a problem instance with sparse ß in dimension
d=100 (sparsity d_nonzero=5). The design matrix X comprises n=100 samples generated from an i.i.d standard normal
distribution. The dimension of the constraint matrix C is d x k matrix. The noise level is σ=0.5. 
The input `zerosum=True` implies that C is the all-ones vector and Cß=0. The n-dimensional outcome vector y
and the regression vector ß is then generated to satisfy the given constraints. 
One can then see the parameters that should be selected.


.. code-block:: default


    m, d, d_nonzero, k, sigma = 100, 200, 5, 1, 0.5
    (X, C, y), sol = random_data(m, d, d_nonzero, k, sigma, zerosum=True, seed=1)
    print(np.nonzero(sol))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (array([  7,  63, 148, 164, 168]),)




Define the classo instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next we can define a default c-lasso problem instance with the generated data:


.. code-block:: default


    problem = classo_problem(X, y, C) 









Change the parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's see some example of change in the parameters


.. code-block:: default


    problem.formulation.huber                   = True
    problem.formulation.concomitant             = False
    problem.model_selection.CV                  = True
    problem.model_selection.LAMfixed            = True
    problem.model_selection.PATH                = True
    problem.model_selection.StabSelparameters.method = 'max'
    problem.model_selection.CVparameters.seed = 1
    problem.model_selection.LAMfixedparameters.rescaled_lam = True
    problem.model_selection.LAMfixedparameters.lam = .1










Check parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can look at the generated problem instance by typing:


.. code-block:: default


    print(problem)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

 
 
    FORMULATION: R2
 
    MODEL SELECTION COMPUTED:  
         Lambda fixed
         Path
         Cross Validation
         Stability selection
 
    LAMBDA FIXED PARAMETERS: 
         numerical_method = not specified
         rescaled lam : True
         threshold : average of the absolute value of beta
         lam = 0.1
 
    PATH PARAMETERS: 
         numerical_method : not specified
         lamin = 0.001
         Nlam = 80
     with log-scale
 
    CROSS VALIDATION PARAMETERS: 
         numerical_method : not specified
         one-SE method : True
         Nsubset = 5
         lamin = 0.001
         Nlam = 80
     with log-scale
 
    STABILITY SELECTION PARAMETERS: 
         numerical_method : not specified
         method : max
         B = 50
         q = 10
         percent_nS = 0.5
         threshold = 0.7
         lamin = 0.01
         Nlam = 50





Solve optimization problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 We only use stability selection as default model selection strategy. 
The command also allows you to inspect the computed stability profile for all variables 
at the theoretical λ


.. code-block:: default


    problem.solve()








Visualisation
^^^^^^^^^^^^^^^

After completion, the results of the optimization and model selection routines 
can be visualized using


.. code-block:: default


    print(problem.solution)


.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_advanced_example_001.png
          :alt: Coefficients at $\lambda$ = 0.1
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_advanced_example_002.png
          :alt: Coefficients across $\lambda$-path using R2
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_advanced_example_003.png
          :alt:  
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_advanced_example_004.png
          :alt: Refitted coefficients after CV model selection
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_advanced_example_005.png
          :alt: Stability selection profile of type max using R2
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_advanced_example_006.png
          :alt: Refitted coefficients after stability selection
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none


     LAMBDA FIXED : 
       Selected variables :  7    63    148    164    168    
       Running time :  0.09s

     PATH COMPUTATION : 
       Running time :  0.432s

     CROSS VALIDATION : 
       Selected variables :  7    10    63    101    148    164    168    
       Running time :  1.767s

     STABILITY SELECTION : 
       Selected variables :  7    63    148    164    168    
       Running time :  5.272s






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  8.888 seconds)


.. _sphx_glr_download_auto_examples_plot_advanced_example.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_advanced_example.py <plot_advanced_example.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_advanced_example.ipynb <plot_advanced_example.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
