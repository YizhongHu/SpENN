Documentation Development
=========================

The documentation source lives in ``docs/`` and is built with Sphinx. Public
Python docstrings should follow the NumPyDoc convention already used in the
package.

Local Build
-----------

Build the HTML documentation:

.. code-block:: bash

   uv run --group docs sphinx-build -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` directly in a browser, or serve the built
site locally:

.. code-block:: bash

   uv run python -m http.server --directory docs/_build/html 8000

The ``docs`` dependency group is opt-in. Normal ``uv sync`` and package use do
not install Sphinx or Numpydoc unless this group is requested.

External inventory links are disabled by default so cluster-local builds do not
need network access. Set ``SPENN_DOCS_INTERSPHINX=1`` when building if you want
Sphinx to link Python, NumPy, and PyTorch objects to their upstream docs.

Website Build
-------------

The separate GitHub Actions workflow in ``.github/workflows/docs.yml`` builds
documentation for pull requests and branch pushes. On the repository default
branch, it uploads the built HTML to GitHub Pages when Pages is configured to
use GitHub Actions.
