==========================
Contributing to c-lasso
==========================

``c-lasso`` is a package that always can be improved. Any feedback can
help a lot to fix some bug and to add possible new functionality.

One can contribute either by reporting  an error, either 


Reporting errors
================

Any errors or general problems can be reported on GitHub's Issue tracker:

   https://github.com/Leo-Simpson/c-lasso/issues

The quickest way resolve a problem is to go through the following steps:

* Have I tested this on the latest GitHub (``master``) version?
To see which version you use, you can run on python :

     >>> import classo
     >>> classo.__version__
  


* Have I provided a sample code block which reproduces the error?  Have I
  tested the code block?

While more information can help, the most important step is to report the
problem, and any missing information can be provided over the course of the
discussion.


Feature requests
================

We recommend opening an issue on `GitHub <https://github.com/Leo-Simpson/c-lasso/issues>`_ to discuss potential changes.

When preparing a feature request, consider providing the following information:

* What problem is this feature trying to solve?

* Is it solvable using Python intrinsics?  How is it currently handled in
  similar modules?

* Can you provide an example code block demonstrating the feature?

* Does this feature require any new dependencies ?







Adding a feature
==================

One can also contribute with a new feature or with fixing a bug.

Feature should be sent as pull requests via `GitHub <https://github.com/Leo-Simpson/c-lasso>`_, specifically to the
``master`` branch, which acts as the main development branch.

Fixes and features are very welcome to ``c-lasso``, and are greatly encouraged.

If you are concerned that a project may not be suitable or may conflict with
ongoing work, then feel free to submit a feature request.

When preparing a pull request, one should check that the code changes:

* Pass existing tests, this can be done by running within the root directory:

  .. code-block:: bash

    $ pip install --upgrade pytest
    $ pytest

* Includes a test case.
  See the files in ``c-lasso/tests`` for examples
  
* Includes some example of use cases.
  See the files in ``c-lasso/examples`` for examples
  
* Depends on standard library. Any features
  requiring an external dependency should only be enabled when the dependenc is available.
  
* Be properly documented. 
  c-lasso's documentation (including docstring in code) uses ReStructuredText format,
  see `Sphinx documentation <http://www.sphinx-doc.org/en/master/>`_ to learn more about editing them. The code
  follows the `NumPy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
  To ensure that documentation is rendered correctly,
  the best bet is to follow the existing examples for function docstrings.
  If you want to test the documentation locally,
  you will need to run the following command lines within the ``c-lasso/docs`` directory :

  .. code-block:: bash

    $ pip install --upgrade sphinx
    $ make html
  
 
