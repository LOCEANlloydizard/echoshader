###############################################################################
# Sphinx configuration
# Keep notebook execution settings consistent with _config.yml.
###############################################################################

author = "Echoshader Contributors"
copyright = "2026"
html_title = "Echoshader"
language = "en"

exclude_patterns = [
    "**.ipynb_checkpoints",
    ".DS_Store",
    "Thumbs.db",
    "_build",
    "MVBS.zarr/**",
    "MVBS.zip",
]

# Notebook execution
nb_execution_mode = "force"
nb_execution_timeout = 600
nb_execution_allow_errors = False
nb_execution_raise_on_error = True
nb_execution_in_temp = False
nb_execution_excludepatterns = []
nb_output_stderr = "show"

extensions = [
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "myst_nb",
    "jupyter_book",
    "sphinx_thebe",
    "sphinx_comments",
    "sphinx_external_toc",
    "sphinx.ext.intersphinx",
    "sphinx_automodapi.automodapi",
    "numpydoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinxcontrib.mermaid",
    "sphinx_jupyterbook_latex",
]

# Navigation
external_toc_path = "_toc.yml"
external_toc_exclude_missing = False
use_multitoc_numbering = True

# HTML appearance and repository links
html_theme = "sphinx_book_theme"
html_baseurl = ""
html_favicon = ""
html_logo = ""
html_sourcelink_suffix = ""

html_theme_options = {
    "search_bar_text": "Search this book...",
    "launch_buttons": {
        "notebook_interface": "classic",
        "binderhub_url": "https://mybinder.org",
        "jupyterhub_url": "",
        "thebe": False,
        "colab_url": "",
    },
    "path_to_docs": "docs/source",
    "repository_url": "https://github.com/echostack-org/echoshader",
    "repository_branch": "main",
    "google_analytics_id": "",
    "extra_navbar": 'Powered by <a href="https://jupyterbook.org">Jupyter Book</a>',
    "extra_footer": "",
    "home_page_in_toc": False,
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": True,
}

comments_config = {
    "hypothesis": False,
    "utterances": False,
}

# Markdown
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "linkify",
    "substitution",
    "tasklist",
]
myst_url_schemes = ["mailto", "http", "https"]
suppress_warnings = ["myst.domains"]

# Figures and code
numfig = True
pygments_style = "sphinx"

# LaTeX
latex_engine = "pdflatex"
use_jupyterbook_latex = True