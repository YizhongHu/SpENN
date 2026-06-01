SPHINXBUILD ?= uv run --group docs sphinx-build
SOURCEDIR = .
BUILDDIR = _build
PORT ?= 8000

.PHONY: html clean serve

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf "$(BUILDDIR)"

serve: html
	uv run python -m http.server --directory "$(BUILDDIR)/html" "$(PORT)"
