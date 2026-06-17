# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = 'PyHDC'
copyright = '2025, GNPower'
author = 'GNPower'
release = '2.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
