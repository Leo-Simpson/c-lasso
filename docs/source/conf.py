# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import sphinx_gallery 
import sys
import warnings

warnings.filterwarnings("ignore")

# -- Project information -----------------------------------------------------

project = 'classo'
copyright = '2020, Leo Simpson'
author = 'Leo Simpson'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
                'sphinx.ext.intersphinx',
                'sphinx.ext.doctest',
                'sphinx.ext.coverage',
                'sphinx.ext.mathjax',
                'sphinx.ext.napoleon',
                'sphinx.ext.autosummary',
                'sphinx_gallery.gen_gallery',]

autosummary_generate = True

def skip(app, what, name, obj, would_skip, options):
    if name == "__init__" or name == "__new__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None)
}

sphinx_gallery_conf = {
    'doc_module': 'numpy',
    'examples_dirs': '../../examples',
    'ignore_pattern': r'/example_',
    'gallery_dirs': 'auto_examples',
}
