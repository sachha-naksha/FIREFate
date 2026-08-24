"""Sphinx configuration for FIREFate."""

from __future__ import annotations

import sys
from pathlib import Path

# Source package lives in ../src (setuptools src layout).
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "src"))

project = "FIREFate"
copyright = "2026, Akanksha Sachan"
author = "Akanksha Sachan"

try:
    from firefate import __version__ as release
except ImportError:
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    # Renders the notebooks pulled in from the firefate_notebooks submodule.
    "myst_nb",
]

# -- Notebooks ---------------------------------------------------------------
# The notebooks live in their own repository (sachha-naksha/firefate_notebooks),
# checked out as a submodule at docs/notebooks. They are research records run on a
# cluster against data that is not distributed, so they are NEVER executed here --
# Sphinx renders the outputs stored in the .ipynb files as committed.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath", "html_image"]
myst_heading_anchors = 3

# Cell outputs from the trajectory notebooks are large; do not truncate them silently.
nb_output_stderr = "remove"

# Dictys is required at runtime for episodic workflows but is not a declared pip dependency;
# mock it so API docs build on Read the Docs without extra installs.
autodoc_mock_imports = ["dictys"]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # The submodule's own front matter, not part of this site.
    "notebooks/README.md",
]

html_theme = "furo"
html_title = f"{project} {release}"
html_static_path = ["_static"]

nitpicky = False

# Optional typing-only imports / upstream types not shipped with docs builds.
nitpick_ignore = [
    ("py:class", "dictys.net.dynamic_network.dynamic_network"),
    ("py:class", "dynamic_network"),
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

copybutton_prompt_text = r">>> |\.\.\. "

# The notebooks submodule is absent in a shallow clone or a lint-only build; without
# it every `notebooks/...` toctree entry becomes a hard error.
if not (Path(__file__).parent / "notebooks" / "index.md").is_file():
    exclude_patterns.append("notebooks/*")
    print(
        "conf.py: docs/notebooks is empty -- building API docs only. "
        "Run `git submodule update --init docs/notebooks` to include the notebooks."
    )
