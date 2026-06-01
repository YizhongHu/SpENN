"""Sphinx configuration for the SpENN documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spenn import __version__  # noqa: E402

project = "SpENN"
author = "SpENN contributors"
copyright = "2026, SpENN contributors"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "numpydoc",
]

if os.environ.get("SPENN_DOCS_INTERSPHINX") == "1":
    extensions.append("sphinx.ext.intersphinx")

autosummary_generate = True
templates_path = ["_templates"]
autodoc_default_options = {
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.nn.parameter",
    "wandb",
]
autodoc_typehints = "none"
autodoc_preserve_defaults = True
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = (
    {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "torch": ("https://docs.pytorch.org/docs/stable/", None),
    }
    if "sphinx.ext.intersphinx" in extensions
    else {}
)

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
html_title = "SpENN"
