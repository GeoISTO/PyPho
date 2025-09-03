# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# help Sphinx to find pypho's source
import sys
from pathlib import Path
sys.path.insert(0, str(Path('..', '..').resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyPho'
copyright = '2025, Gautier LAURENT'
author = 'Gautier LAURENT'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration', # timing of the building of different pages
    'sphinx.ext.autodoc', # auto generate the code doc
    'sphinx.ext.autosummary', # auto generate the table of content of code
    "pyvista.ext.plot_directive", # these three are for pyvista
    "pyvista.ext.viewer_directive",
    "sphinx_design",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
