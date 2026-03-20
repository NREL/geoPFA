# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as get_version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Geothermal Play Fairway Analysis"
copyright = "2025, Nicole Taverna"
author = "Nicole Taverna"

release = get_version("geopfa").split("+")[0]
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Extension configuration -------------------------------------------------

# -- Autodoc configuration --
autoclass_content = "both"          # Merge __init__ docstring into the class page
autodoc_member_order = "bysource"   # Keep methods in source-code order
autodoc_inherit_docstrings = True   # Inherit docstrings from base classes
autodoc_typehints = "none"
add_module_names = False            # Drop "geopfa." prefix from signatures

# -- Autosummary configuration --
autosummary_generate = True                   # Auto-generate stub pages
autosummary_generate_overwrite = True         # Regenerate stubs on every build
autosummary_imported_members = False          # Skip re-exported names

# -- BibTeX configuration --
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

# -- Intersphinx configuration --
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- Napoleon configuration --
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- MyST Parser configuration --
myst_enable_extensions = [
    "dollarmath",
    "fieldlist",
    "substitution",
    "tasklist",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = f"Geothermal PFA {release}"
html_static_path = ["_static"]
