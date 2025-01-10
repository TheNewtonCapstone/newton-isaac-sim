# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "newton-isaac-sim"
copyright = "2025, TheNewtonCapstone"
author = "TheNewtonCapstone"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "breathe",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"

# -- Breathe Configuration ---------------------------------------------------
# https://breathe.readthedocs.io/en/latest/quickstart.html#configuration

breathe_projects = {"newton-isaac-sim": "doxygen/xml"}
breathe_default_project = "newton-isaac-sim"

import subprocess

# Generate Doxygen XML
subprocess.call("doxygen Doxyfile", shell=True)

# Generate Sphinx pages
subprocess.call("python generate_rst.py", shell=True)
