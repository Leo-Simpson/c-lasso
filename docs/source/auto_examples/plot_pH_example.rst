.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_pH_example.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_pH_example.py:


pH prediction using the Central Park soil dataset 
===================================================

The next microbiome example considers the 
`Central Park Soil dataset <https://github.com/Leo-Simpson/c-lasso/tree/master/examples/pH_data>`_ from
`Ramirez et al. <https://royalsocietypublishing.org/doi/full/10.1098/rspb.2014.1988>`_. The sample locations are shown in the Figure on the right.
The task is to predict pH concentration in the soil from microbial abundance data. This task was also considered in `Tree-Aggregated Predictive Modeling of Microbiome Data <https://www.biorxiv.org/content/10.1101/2020.09.01.277632v1>`_.


.. code-block:: default


    from classo import classo_problem
    import numpy as np
    from copy import deepcopy as dc
    import scipy.io as sio








Load data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default


    pH = sio.loadmat("pH_data/pHData.mat")
    tax = sio.loadmat("pH_data/taxTablepHData.mat")["None"][0]

    X, Y_uncent = pH["X"], pH["Y"].T[0]
    y = Y_uncent - np.mean(Y_uncent)  # Center Y
    print(X.shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (88, 116)




Set up c-lassso problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default


    problem = classo_problem(X, y) 

    problem.model_selection.StabSelparameters.method      = 'lam'
    problem.model_selection.PATH = True
    problem.model_selection.LAMfixed = True
    problem.model_selection.PATHparameters.n_active = X.shape[1] + 1








Solve for R1
^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    problem.formulation.concomitant = False
    problem.solve()
    print(problem, problem.solution)




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_001.png
          :alt: Coefficients at $\lambda$ = 0.218
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_002.png
          :alt: Coefficients across $\lambda$-path using R1
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_003.png
          :alt: Stability selection profile of type lam using R1
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_004.png
          :alt: Refitted coefficients after stability selection
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

 
 
    FORMULATION: R1
 
    MODEL SELECTION COMPUTED:  
         Lambda fixed
         Path
         Stability selection
 
    LAMBDA FIXED PARAMETERS: 
         numerical_method = Path-Alg
         rescaled lam : True
         threshold = 0.008
         lam : theoretical
         theoretical_lam = 0.2182
 
    PATH PARAMETERS: 
         numerical_method : Path-Alg
         lamin = 0.001
         Nlam = 80
 
         maximum active variables = 117
 
    STABILITY SELECTION PARAMETERS: 
         numerical_method : Path-Alg
         method : lam
         B = 50
         q = 10
         percent_nS = 0.5
         threshold = 0.7
         lam = theoretical
         theoretical_lam = 0.3085
 
     LAMBDA FIXED : 
       Selected variables :  18    19    39    43    62    85    93    94    102    107    
       Running time :  0.013s

     PATH COMPUTATION : 
       Running time :  0.204s

     STABILITY SELECTION : 
       Selected variables :  19    62    94    
       Running time :  0.429s





Solve for R2
^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    problem.formulation.huber = True
    problem.solve()
    print(problem, problem.solution)





.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_005.png
          :alt: Coefficients at $\lambda$ = 0.218
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_006.png
          :alt: Coefficients across $\lambda$-path using R2
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_007.png
          :alt: Stability selection profile of type lam using R2
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_008.png
          :alt: Refitted coefficients after stability selection
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

 
 
    FORMULATION: R2
 
    MODEL SELECTION COMPUTED:  
         Lambda fixed
         Path
         Stability selection
 
    LAMBDA FIXED PARAMETERS: 
         numerical_method = Path-Alg
         rescaled lam : True
         threshold = 0.008
         lam : theoretical
         theoretical_lam = 0.2182
 
    PATH PARAMETERS: 
         numerical_method : Path-Alg
         lamin = 0.001
         Nlam = 80
 
         maximum active variables = 117
 
    STABILITY SELECTION PARAMETERS: 
         numerical_method : Path-Alg
         method : lam
         B = 50
         q = 10
         percent_nS = 0.5
         threshold = 0.7
         lam = theoretical
         theoretical_lam = 0.3085
 
     LAMBDA FIXED : 
       Selected variables :  18    19    39    43    57    62    85    93    94    107    
       Running time :  0.066s

     PATH COMPUTATION : 
       Running time :  0.358s

     STABILITY SELECTION : 
       Selected variables :  19    62    94    
       Running time :  1.37s





Solve for R3
^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    problem.formulation.concomitant = True
    problem.formulation.huber = False
    problem.solve()
    print(problem, problem.solution)





.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_009.png
          :alt: Coefficients at $\lambda$ = 0.218
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_010.png
          :alt: Coefficients across $\lambda$-path using R3
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_011.png
          :alt: Scale estimate across $\lambda$-path using R3
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_012.png
          :alt: Stability selection profile of type lam using R3
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_013.png
          :alt: Refitted coefficients after stability selection
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

 
 
    FORMULATION: R3
 
    MODEL SELECTION COMPUTED:  
         Lambda fixed
         Path
         Stability selection
 
    LAMBDA FIXED PARAMETERS: 
         numerical_method = Path-Alg
         rescaled lam : True
         threshold = 0.008
         lam : theoretical
         theoretical_lam = 0.2182
 
    PATH PARAMETERS: 
         numerical_method : Path-Alg
         lamin = 0.001
         Nlam = 80
 
         maximum active variables = 117
 
    STABILITY SELECTION PARAMETERS: 
         numerical_method : Path-Alg
         method : lam
         B = 50
         q = 10
         percent_nS = 0.5
         threshold = 0.7
         lam = theoretical
         theoretical_lam = 0.3085
 
     LAMBDA FIXED : 
       Sigma  =  0.633
       Selected variables :  15    18    19    23    25    27    43    47    50    53    57    58    62    89    93    94    104    107    
       Running time :  0.032s

     PATH COMPUTATION : 
       Running time :  0.23s

     STABILITY SELECTION : 
       Selected variables :  18    19    43    62    94    107    
       Running time :  1.002s





Solve for R4
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Remark : we reset the numerical method here, 
because it has been automatically set to '¨Path-Alg'
for previous computations, but for R4, "DR" is much better
as explained in the documentation, R4 "Path-Alg" is a method for fixed lambda
but is (paradoxically) bad to compute the lambda-path 
because of the absence of possible warm-start in this method


.. code-block:: default


    problem.model_selection.PATHparameters.numerical_method = "DR"
    problem.formulation.huber = True
    problem.solve()
    print(problem, problem.solution)





.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_014.png
          :alt: Coefficients at $\lambda$ = 0.218
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_015.png
          :alt: Coefficients across $\lambda$-path using R4
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_016.png
          :alt: Scale estimate across $\lambda$-path using R4
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_017.png
          :alt: Stability selection profile of type lam using R4
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_pH_example_018.png
          :alt: Refitted coefficients after stability selection
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

 
 
    FORMULATION: R4
 
    MODEL SELECTION COMPUTED:  
         Lambda fixed
         Path
         Stability selection
 
    LAMBDA FIXED PARAMETERS: 
         numerical_method = Path-Alg
         rescaled lam : True
         threshold = 0.008
         lam : theoretical
         theoretical_lam = 0.2182
 
    PATH PARAMETERS: 
         numerical_method : DR
         lamin = 0.001
         Nlam = 80
 
         maximum active variables = 117
 
    STABILITY SELECTION PARAMETERS: 
         numerical_method : Path-Alg
         method : lam
         B = 50
         q = 10
         percent_nS = 0.5
         threshold = 0.7
         lam = theoretical
         theoretical_lam = 0.3085
 
     LAMBDA FIXED : 
       Sigma  =  0.284
       Selected variables :  15    18    19    23    27    43    47    50    53    57    58    62    89    93    94    104    107    
       Running time :  0.052s

     PATH COMPUTATION : 
       Running time :  79.059s

     STABILITY SELECTION : 
       Selected variables :  18    19    43    62    94    107    
       Running time :  1.474s






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  29.420 seconds)


.. _sphx_glr_download_auto_examples_plot_pH_example.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_pH_example.py <plot_pH_example.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_pH_example.ipynb <plot_pH_example.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
