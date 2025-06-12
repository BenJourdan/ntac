"""Sphinx configuration file for the ntac documentation."""
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))



# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ntac"
copyright = "2025, Ben Jourdan, Gregory Schwartzman, Arie Matsliah"
author = "Ben Jourdan, Gregory Schwartzman, Arie Matsliah"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",      # only if you use NumPy or Google style docstrings
    "sphinx.ext.autosummary",   # optional: generates summary tables
    "sphinx_autodoc_typehints", # optional: cleaner type hints
]

autosummary_generate = True     # optionally auto-generate .rst stubs

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = [
    "_static"
    ]

html_css_files = [
    'custom.css',
]