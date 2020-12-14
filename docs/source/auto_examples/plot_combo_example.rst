.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_combo_example.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_combo_example.py:


BMI prediction using the COMBO dataset 
==========================================

We first consider the `COMBO data set <https://github.com/Leo-Simpson/c-lasso/tree/master/examples/COMBO_data>`_
and show how to predict Body Mass Index (BMI) from microbial genus abundances and two non-compositional covariates  using "filtered_data".


.. code-block:: default


    from classo import csv_to_np, classo_problem, clr
    import numpy as np








Load microbiome and covariate data X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default


    X0  = csv_to_np('COMBO_data/complete_data/GeneraCounts.csv', begin = 0).astype(float)
    X_C = csv_to_np('COMBO_data/CaloriData.csv', begin = 0).astype(float)
    X_F = csv_to_np('COMBO_data/FatData.csv', begin = 0).astype(float)








Load BMI measurements y
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    y   = csv_to_np('COMBO_data/BMI.csv', begin = 0).astype(float)[:, 0]
    labels = csv_to_np('COMBO_data/complete_data/GeneraPhylo.csv').astype(str)[:, -1]









Normalize/transform data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    y   = y - np.mean(y) #BMI data (n = 96)
    X_C = X_C - np.mean(X_C, axis = 0)  #Covariate data (Calorie)
    X_F = X_F - np.mean(X_F, axis = 0)  #Covariate data (Fat)
    X0 = clr(X0, 1 / 2).T








Set up design matrix and zero-sum constraints for 45 genera
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default


    X     = np.concatenate((X0, X_C, X_F, np.ones((len(X0), 1))), axis = 1) # Joint microbiome and covariate data and offset
    label = np.concatenate([labels, np.array(['Calorie', 'Fat', 'Bias'])])
    C = np.ones((1, len(X[0])))
    C[0, -1], C[0, -2], C[0, -3] = 0., 0., 0.











Set up c-lassso problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default


    problem = classo_problem(X, y, C) 








Use stability selection with theoretical lambda [Combettes & Müller, 2020b]


.. code-block:: default

    problem.model_selection.StabSelparameters.method      = 'lam'
    problem.model_selection.StabSelparameters.threshold_label = 0.5








Use formulation R3
^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    problem.formulation.concomitant = True

    problem.solve()
    print(problem)
    print(problem.solution)




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_combo_example_001.png
          :alt: Stability selection profile of type lam using R3
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_combo_example_002.png
          :alt: Refitted coefficients after stability selection
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

 
 
    FORMULATION: R3
 
    MODEL SELECTION COMPUTED:  
         Stability selection
 
    STABILITY SELECTION PARAMETERS: 
         numerical_method : Path-Alg
         method : lam
         B = 50
         q = 10
         percent_nS = 0.5
         threshold = 0.7
         lam = theoretical
         theoretical_lam = 0.2824


     STABILITY SELECTION : 
       Selected variables :  27    56    
       Running time :  0.571s





Use formulation R4
^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    problem.formulation.huber = True
    problem.formulation.concomitant = True

    problem.solve()
    print(problem)
    print(problem.solution)



.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_combo_example_003.png
          :alt: Stability selection profile of type lam using R4
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_combo_example_004.png
          :alt: Refitted coefficients after stability selection
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

 
 
    FORMULATION: R4
 
    MODEL SELECTION COMPUTED:  
         Stability selection
 
    STABILITY SELECTION PARAMETERS: 
         numerical_method : Path-Alg
         method : lam
         B = 50
         q = 10
         percent_nS = 0.5
         threshold = 0.7
         lam = theoretical
         theoretical_lam = 0.2824


     STABILITY SELECTION : 
       Selected variables :  27    56    
       Running time :  0.866s






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  2.153 seconds)


.. _sphx_glr_download_auto_examples_plot_combo_example.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_combo_example.py <plot_combo_example.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_combo_example.ipynb <plot_combo_example.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
