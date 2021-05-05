.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_CentralParkSoil.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_CentralParkSoil.py:


pH prediction using the Central Park soil dataset 
=========================================================


The next microbiome example considers the [Central Park Soil dataset](./examples/CentralParkSoil) from [Ramirez et al.](https://royalsocietypublishing.org/doi/full/10.1098/rspb.2014.1988). The sample locations are shown in the Figure on the right.)

The task is to predict pH concentration in the soil from microbial abundance data.

This task is also done in `Tree-Aggregated Predictive Modeling of Microbiome Data <https://www.biorxiv.org/content/10.1101/2020.09.01.277632v1>`_.


.. code-block:: default


    from classo import classo_problem
    import matplotlib.pyplot as plt
    import numpy as np








Load data
^^^^^^^^^^^^^^^^^^^


.. code-block:: default


    data = np.load('CentralParkSoil/cps.npz')

    x = data["x"]
    label = data["label"]
    y = data["y"]

    A = np.load('CentralParkSoil/A.npy')








Preprocess: taxonomy aggregation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default


    label_short = np.array([l.split("::")[-1] for l in label])

    pseudo_count = 1
    X = np.log(pseudo_count+x)
    nleaves = np.sum(A,axis = 0)
    logGeom = X.dot(A)/nleaves

    n,d = logGeom.shape

    tr = np.random.permutation(n)[:int(0.8*n)]








Cross validation and Path Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default


    problem = classo_problem(logGeom[tr], y[tr], label = label_short)

    problem.formulation.w = 1/nleaves
    problem.formulation.intercept     = True
    problem.formulation.concomitant = False

    problem.model_selection.StabSel   = False
    problem.model_selection.PATH   = True
    problem.model_selection.CV   = True
    problem.model_selection.CVparameters.seed = 6 # one could change logscale, Nsubset, oneSE
    print(problem)

    problem.solve()
    print(problem.solution)

    selection = problem.solution.CV.selected_param[1:] # exclude the intercept
    print(label[selection])




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_CentralParkSoil_001.png
          :alt: Coefficients across $\lambda$-path using R1
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_CentralParkSoil_002.png
          :alt:  
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_CentralParkSoil_003.png
          :alt: Refitted coefficients after CV model selection
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

 
 
    FORMULATION: R1
 
    MODEL SELECTION COMPUTED:  
         Path
         Cross Validation
 
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


     PATH COMPUTATION : 
     There is also an intercept.  
       Running time :  4.572s

     CROSS VALIDATION : 
     Intercept : 5.914250472305105
       Selected variables :  p__Bacteroidetes    o__Acidobacteriales    k__Bacteria    
       Running time :  23.638s

    ['Life::k__Bacteria::p__Bacteroidetes'
     'Life::k__Bacteria::p__Acidobacteria::c__Acidobacteriia::o__Acidobacteriales'
     'Life::k__Bacteria']




Prediction plot
""""""""""""""""""""


.. code-block:: default


    te = np.array([i for i in range(len(y)) if not i in tr])
    alpha = problem.solution.CV.refit
    yhat = logGeom[te].dot(alpha[1:])+alpha[0]

    M1, M2 = max(y[te]), min(y[te])
    plt.plot(yhat, y[te], 'bo', label = 'sample of the testing set')
    plt.plot([M1, M2], [M1, M2], 'k-', label = "identity")
    plt.xlabel('predictor yhat'), plt.ylabel('real y'), plt.legend()
    plt.tight_layout()




.. image:: /auto_examples/images/sphx_glr_plot_CentralParkSoil_004.png
    :alt: plot CentralParkSoil
    :class: sphx-glr-single-img





Stability selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default


    problem = classo_problem(logGeom[tr], y[tr], label = label_short)

    problem.formulation.w = 1/nleaves
    problem.formulation.intercept     = True
    problem.formulation.concomitant = False


    problem.model_selection.PATH   = False
    problem.model_selection.CV   = False
    # can change q, B, nS, method, threshold etc in problem.model_selection.StabSelparameters

    problem.solve()

    print(problem, problem.solution)

    selection = problem.solution.StabSel.selected_param[1:] # exclude the intercept
    print(label[selection])




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_CentralParkSoil_005.png
          :alt: Stability selection profile of type first using R1
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_CentralParkSoil_006.png
          :alt: Refitted coefficients after stability selection
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

 
 
    FORMULATION: R1
 
    MODEL SELECTION COMPUTED:  
         Stability selection
 
    STABILITY SELECTION PARAMETERS: 
         numerical_method : Path-Alg
         method : first
         B = 50
         q = 10
         percent_nS = 0.5
         threshold = 0.7
         lamin = 0.01
         Nlam = 50
 
     STABILITY SELECTION : 
       Selected variables :  intercept    p__Bacteroidetes    o__Acidobacteriales    c__Acidobacteria-6    k__Bacteria    
       Running time :  55.132s

    ['Life::k__Bacteria::p__Bacteroidetes'
     'Life::k__Bacteria::p__Acidobacteria::c__Acidobacteriia::o__Acidobacteriales'
     'Life::k__Bacteria::p__Acidobacteria::c__Acidobacteria-6'
     'Life::k__Bacteria']




Prediction plot
""""""""""""""""""""


.. code-block:: default


    te = np.array([i for i in range(len(y)) if not i in tr])
    alpha = problem.solution.StabSel.refit
    yhat = logGeom[te].dot(alpha[1:])+alpha[0]

    M1, M2 = max(y[te]), min(y[te])
    plt.plot(yhat, y[te], 'bo', label = 'sample of the testing set')
    plt.plot([M1, M2],[M1, M2], 'k-', label = "identity")
    plt.xlabel('predictor yhat'), plt.ylabel('real y'), plt.legend()
    plt.tight_layout()


.. image:: /auto_examples/images/sphx_glr_plot_CentralParkSoil_007.png
    :alt: plot CentralParkSoil
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  25.884 seconds)


.. _sphx_glr_download_auto_examples_plot_CentralParkSoil.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_CentralParkSoil.py <plot_CentralParkSoil.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_CentralParkSoil.ipynb <plot_CentralParkSoil.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
