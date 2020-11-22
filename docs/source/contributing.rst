==========================
Contributing to c-lasso
==========================

``c-lasso`` is a package that always can be improved. Any feedback can
help a lot to fix some bug and to add possible new functionality.


Development Portal
==================

Code development is currently hosted at GitHub.  Issues, feature requests, and
code contributions are currently handled there.

   https://github.com/Leo-Simpson/c-lasso/


Reporting errors
================

Any errors or general problems can be reported on GitHub's Issue tracker:

   https://github.com/Leo-Simpson/c-lasso/issues

The quickest way resolve a problem is to go through the following steps:

* Have I tested this on the latest GitHub (``master``) version?

* Have I provided a sample code block which reproduces the error?  Have I
  tested the code block?

* Have I included the necessary input or output files?

  Sometimes a file attachment is required, since uncommon whitespace or
  Unicode characters may be missing from a standard cut-and-paste of the file.

* Have I provided a backtrace from the error?

Usually this is enough information to reproduce and resolve the problem.  In
some cases, we may need more information about your system, including the
following:

* Your ``classo`` version::

     >>> import classo
     >>> classo.__version__

* The version and build of python::

     >>> import sys
     >>> print(sys.version)

* Your operating system (Linux, OS X, Windows, etc.), and your distribution if
  relevant.

While more information can help, the most important step is to report the
problem, and any missing information can be provided over the course of the
discussion.


Feature requests
================

Feature requests are welcome, and can be submitted as Issues in the GitHub
tracker.

   https://github.com/Leo-Simpson/c-lasso/issues

When preparing a feature request, consider providing the following information:

* What problem is this feature trying to solve?

* Is it solvable using Python intrinsics?  How is it currently handled in
  similar modules?

* Can you provide an example code block demonstrating the feature?

* Does this feature require any new dependencies ?


Contributing to c-lasso
======================

Fixes and features are very welcome to ``c-lasso``, and are greatly encouraged.

If you are concerned that a project may not be suitable or may conflict with
ongoing work, then feel free to submit a feature request with comment noting
that you are happy to provide the feature.

Feature should be sent as pull requests via GitHub, specifically to the
``master`` branch, which acts as the main development branch.

Explicit patches via email are also welcome.

When preparing a pull request, consider the following advice:

* Code changes must pass existing tests::

     $ pytest

* Providing a test case for your example would be greatly appreciated.  See
  the test files in ``tests``for examples.
  
* Changes must be accompanied by **updated documentation** and examples.

* Features should generally only depend on the standard library.  Any features
  requiring an external dependency should only be enabled when the dependency
  is available.
  
  
  
Checking and building documentation
-----------------------------------

c-lasso's documentation (including docstring in code) uses ReStructuredText format,
see `Sphinx documentation <http://www.sphinx-doc.org/en/master/>`_ to learn more about editing them. The code
follows the `NumPy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html>`_.


All changes to the codebase must be properly documented. To ensure that documentation is rendered correctly, the best bet is to follow the existing examples for function docstrings. If you want to test the documentation locally, you will need to install the following package:

.. code-block:: bash

  $ pip install --upgrade sphinx

and then within the ``c-lasso/docs`` directory do:

.. code-block:: bash

  $ make html

