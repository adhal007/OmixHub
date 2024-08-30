import os
import sys

# Add the 'src' directory to sys.path
sys.path.insert(0, os.path.abspath('../'))

# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'OmixHub'
copyright = '2024, Abhilash Dhal'
author = 'Abhilash Dhal'

# The full version, including alpha/beta/rc tags
release = '0.6.4'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For Google style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver'
]
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'blue'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [] 

# -- Options for Napoleon extension ------------------------------------------

# Use Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False


napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
# -- Options for autodoc extension -------------------------------------------

# Add module names to the documentation
add_module_names = True

# Sort members by type
autodoc_member_order = 'bysource'

# -- Options for todo extension ----------------------------------------------

# Include TODOs in the output
# todo_include_todos = True