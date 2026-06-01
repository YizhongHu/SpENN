# SpENN Docs

The documentation source lives in this directory and is built with Sphinx plus
Numpydoc. The generated HTML is not committed; it is written to
`docs/_build/html/`.

## Use Locally

Build the HTML docs from the repository root:

```bash
uv run --group docs sphinx-build -b html docs docs/_build/html
```

For the same warning policy used by CI:

```bash
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` directly, or serve the site:

```bash
uv run python -m http.server --directory docs/_build/html 8000
```

Then visit `http://localhost:8000`.

The Makefile wraps the same workflow:

```bash
make -C docs html
make -C docs serve
```

Remote cluster note: if the server is remote, run the server command there and
forward the port from your laptop, for example:

```bash
ssh -N -L 8000:localhost:8000 <user>@<login-host>
```

Then open `http://localhost:8000` on your laptop.

External inventory links are off by default so local builds work without
network access. To link Python, NumPy, and PyTorch objects to upstream docs:

```bash
SPENN_DOCS_INTERSPHINX=1 uv run --group docs sphinx-build -b html docs docs/_build/html
```

## Deploy On GitHub Pages

This repo includes `.github/workflows/docs.yml`. The workflow builds the docs on
pull requests and pushes that touch docs, package source, `pyproject.toml`, or
the workflow itself. It deploys only when the workflow runs on the repository
default branch.

One-time repository setup:

1. Commit and push `docs/`, `.github/workflows/docs.yml`, `pyproject.toml`, and
   `uv.lock`.
2. In GitHub, open the repository settings.
3. Go to **Pages**.
4. Under **Build and deployment**, set **Source** to **GitHub Actions**.

Deploy:

```bash
git push origin <default-branch>
```

GitHub Actions will build the docs, upload `docs/_build/html`, and deploy the
artifact to Pages. You can also run the `docs` workflow manually from the
Actions tab on the default branch.

The Pages URL is shown in the workflow's `deploy` job and in the repository's
Pages settings. For a normal project site, it is usually:

```text
https://<owner>.github.io/<repository>/
```
