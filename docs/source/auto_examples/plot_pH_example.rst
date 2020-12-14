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
    metadata = pd.read_table('originals/88soils_modified_metadata.txt', index_col=0)
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


.. code-block:: pytb

    Traceback (most recent call last):
      File "/Users/lsimpson/Desktop/GitHub/c-lasso/examples/plot_pH_example.py", line 33, in <module>
        metadata = pd.read_table('originals/88soils_modified_metadata.txt', index_col=0)
      File "/Users/lsimpson/opt/anaconda3/envs/myenv/lib/python3.9/site-packages/pandas/io/parsers.py", line 767, in read_table
        return read_csv(**locals())
      File "/Users/lsimpson/opt/anaconda3/envs/myenv/lib/python3.9/site-packages/pandas/io/parsers.py", line 688, in read_csv
        return _read(filepath_or_buffer, kwds)
      File "/Users/lsimpson/opt/anaconda3/envs/myenv/lib/python3.9/site-packages/pandas/io/parsers.py", line 454, in _read
        parser = TextFileReader(fp_or_buf, **kwds)
      File "/Users/lsimpson/opt/anaconda3/envs/myenv/lib/python3.9/site-packages/pandas/io/parsers.py", line 948, in __init__
        self._make_engine(self.engine)
      File "/Users/lsimpson/opt/anaconda3/envs/myenv/lib/python3.9/site-packages/pandas/io/parsers.py", line 1180, in _make_engine
        self._engine = CParserWrapper(self.f, **self.options)
      File "/Users/lsimpson/opt/anaconda3/envs/myenv/lib/python3.9/site-packages/pandas/io/parsers.py", line 2010, in __init__
        self._reader = parsers.TextReader(src, **kwds)
      File "pandas/_libs/parsers.pyx", line 382, in pandas._libs.parsers.TextReader.__cinit__
      File "pandas/_libs/parsers.pyx", line 674, in pandas._libs.parsers.TextReader._setup_parser_source
    FileNotFoundError: [Errno 2] No such file or directory: 'originals/88soils_modified_metadata.txt'




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


Solve for R2
^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    problem.formulation.huber = True
    problem.solve()
    print(problem, problem.solution)



Solve for R3
^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: default

    problem.formulation.concomitant = True
    problem.formulation.huber = False
    problem.solve()
    print(problem, problem.solution)



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




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.083 seconds)


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
