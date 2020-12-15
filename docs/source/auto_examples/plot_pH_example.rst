.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_pH_example.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_pH_example.py:


pH prediction using the 88 soils dataset 
===================================================

The next microbiome example considers a
`Soil dataset <https://github.com/Leo-Simpson/c-lasso/tree/master/examples/pH_data>`_ .

The data are generated thanks to a `qiime2 workflow <https://github.com/Leo-Simpson/c-lasso/blob/master/examples/pH_data/qiime2/read%20data.ipynb>`_
similar to `a gneiss tutorial <https://github.com/biocore/gneiss/blob/master/ipynb/88soils/88soils-qiime2-tutorial.ipynb>`_.

This workflow treat `some files <https://github.com/Leo-Simpson/c-lasso/blob/master/examples/pH_data/qiime2/originals>`_ 
taken from `gneiss GitHub <https://github.com/biocore/gneiss/tree/master/ipynb/88soils>`_.


The task is to predict pH concentration in the soil from microbial abundance data.

A similar analysis is also done in `Tree-Aggregated Predictive Modeling of Microbiome Data <https://www.biorxiv.org/content/10.1101/2020.09.01.277632v1>`_.
 `on another dataset <https://royalsocietypublishing.org/doi/full/10.1098/rspb.2014.1988>`_


.. code-block:: default


    from classo import classo_problem
    import numpy as np
    import pandas as pd










Load data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default



    t = pd.read_csv('pH_data/qiime2/news/table.csv', index_col=0)
    metadata = pd.read_table('pH_data/qiime2/originals/88soils_modified_metadata.txt', index_col=0)
    y_uncent = metadata["ph"].values


    X = t.values
    label = t.columns




    # second option to load the data
    # import scipy.io as sio
    # pH = sio.loadmat("pH_data/matlab/pHData.mat")
    # tax = sio.loadmat("pH_data/matlab/taxTablepHData.mat")["None"][0]
    # X, y_uncent = pH["X"], pH["Y"].T[0]
    # label = None

    y = y_uncent - np.mean(y_uncent)  # Center Y
    print(X.shape)
    print(y.shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (89, 118)
    (89,)




Set up c-lassso problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default


    problem = classo_problem(X, y, label = label) 

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
         threshold = 0.001
         lam : theoretical
         theoretical_lam = 0.2176
 
    PATH PARAMETERS: 
         numerical_method : Path-Alg
         lamin = 0.001
         Nlam = 80
 
         maximum active variables = 119
 
    STABILITY SELECTION PARAMETERS: 
         numerical_method : Path-Alg
         method : lam
         B = 50
         q = 10
         percent_nS = 0.5
         threshold = 0.7
         lam = theoretical
         theoretical_lam = 0.3095
 
     LAMBDA FIXED : 
       Selected variables :  y0    y2    y82    
       Running time :  0.009s

     PATH COMPUTATION : 
       Running time :  0.208s

     STABILITY SELECTION : 
       Selected variables :  y0    
       Running time :  0.267s





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
         threshold = 0.001
         lam : theoretical
         theoretical_lam = 0.2176
 
    PATH PARAMETERS: 
         numerical_method : Path-Alg
         lamin = 0.001
         Nlam = 80
 
         maximum active variables = 119
 
    STABILITY SELECTION PARAMETERS: 
         numerical_method : Path-Alg
         method : lam
         B = 50
         q = 10
         percent_nS = 0.5
         threshold = 0.7
         lam = theoretical
         theoretical_lam = 0.3095
 
     LAMBDA FIXED : 
       Selected variables :  y0    y2    y82    
       Running time :  0.049s

     PATH COMPUTATION : 
       Running time :  0.589s

     STABILITY SELECTION : 
       Selected variables :  y0    
       Running time :  1.551s





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
         threshold = 0.001
         lam : theoretical
         theoretical_lam = 0.2176
 
    PATH PARAMETERS: 
         numerical_method : Path-Alg
         lamin = 0.001
         Nlam = 80
 
         maximum active variables = 119
 
    STABILITY SELECTION PARAMETERS: 
         numerical_method : Path-Alg
         method : lam
         B = 50
         q = 10
         percent_nS = 0.5
         threshold = 0.7
         lam = theoretical
         theoretical_lam = 0.3095
 
     LAMBDA FIXED : 
       Sigma  =  1.938
       Selected variables :  y0    y2    y82    
       Running time :  0.03s

     PATH COMPUTATION : 
       Running time :  0.396s

     STABILITY SELECTION : 
       Selected variables :  y0    
       Running time :  0.72s





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
         threshold = 0.001
         lam : theoretical
         theoretical_lam = 0.2176
 
    PATH PARAMETERS: 
         numerical_method : DR
         lamin = 0.001
         Nlam = 80
 
         maximum active variables = 119
 
    STABILITY SELECTION PARAMETERS: 
         numerical_method : Path-Alg
         method : lam
         B = 50
         q = 10
         percent_nS = 0.5
         threshold = 0.7
         lam = theoretical
         theoretical_lam = 0.3095
 
     LAMBDA FIXED : 
       Sigma  =  0.969
       Selected variables :  y0    y2    y82    
       Running time :  0.052s

     PATH COMPUTATION : 
       Running time :  355.96s

     STABILITY SELECTION : 
       Selected variables :  y0    
       Running time :  1.184s






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 6 minutes  7.908 seconds)


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
