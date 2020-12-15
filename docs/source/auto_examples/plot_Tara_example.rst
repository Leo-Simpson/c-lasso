.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_Tara_example.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_Tara_example.py:


Ocean salinity prediction based on marin microbiome data
=========================================================

We repoduce an example of prediction of ocean salinity over ocean microbiome data
that has been introduced in `this article <https://www.biorxiv.org/content/10.1101/2020.09.01.277632v1.full>`_,
where the R package `trac <https://github.com/jacobbien/trac>`_ (which uses c-lasso)
has been used. 

The data come originally from `trac <https://github.com/jacobbien/trac>`_,
then it is preprocessed in python in this `notebook <https://github.com/Leo-Simpson/c-lasso/examples/Tara/preprocess>`_.



Bien, J., Yan, X., Simpson, L. and Müller, C. (2020).
Tree-Aggregated Predictive Modeling of Microbiome Data :

"Integrative marine data collection efforts such as Tara Oceans (Sunagawa et al., 2020)
or the Simons CMAP (https://simonscmap.com)
provide the means to investigate ocean ecosystems on a global scale.
Using Tara’s environmental and microbial survey of ocean surface water (Sunagawa, 2015),
we next illustrate how trac can be used to globally connect environmental covariates
and the ocean microbiome. As an example, we learn a global predictive model of ocean salinity
from n = 136 samples and p = 8916 miTAG OTUs (Logares et al., 2014).
trac identifies four taxonomic aggregations,
the kingdom bacteria and the phylum Bacteroidetes being negatively associated
and the classes Alpha and Gammaproteobacteria being positively associated with marine salinity.


.. code-block:: default


    from classo import classo_problem
    import matplotlib.pyplot as plt
    import numpy as np








Load data
^^^^^^^^^^^^^^^^^^^


.. code-block:: default


    data = np.load('Tara/tara.npz')


    logGeom = data["logGeom"]
    nleaves = data["nleaves"]
    y = data["y"]
    label_nodes = data["label_nodes"]
    label_short = data["label_short"]
    tr = data["tr"]









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
    print(label_nodes[selection])




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_Tara_example_001.png
          :alt: Coefficients across $\lambda$-path using R1
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_Tara_example_002.png
          :alt:  
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_Tara_example_003.png
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
       Running time :  137.147s

     CROSS VALIDATION : 
     Intercept : 34.26188897229179
       Selected variables :  Gammaproteobacteria    Alphaproteobacteria    Bacteria    
       Running time :  710.037s

    ['Life::Bacteria::Proteobacteria::Gammaproteobacteria'
     'Life::Bacteria::Proteobacteria::Alphaproteobacteria' 'Life::Bacteria']




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
    plt.show()




.. image:: /auto_examples/images/sphx_glr_plot_Tara_example_004.png
    :alt: plot Tara example
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
    print(label_nodes[selection])




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_Tara_example_005.png
          :alt: Stability selection profile of type first using R1
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_Tara_example_006.png
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
       Selected variables :  intercept    Bacteroidetes    Alphaproteobacteria    Bacteria    
       Running time :  1484.467s

    ['Life::Bacteria::Bacteroidetes'
     'Life::Bacteria::Proteobacteria::Alphaproteobacteria' 'Life::Bacteria']




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
    plt.show()


.. image:: /auto_examples/images/sphx_glr_plot_Tara_example_007.png
    :alt: plot Tara example
    :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 38 minutes  55.152 seconds)


.. _sphx_glr_download_auto_examples_plot_Tara_example.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_Tara_example.py <plot_Tara_example.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_Tara_example.ipynb <plot_Tara_example.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
