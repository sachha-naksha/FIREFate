# Minimal makefile for Sphinx documentation
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help Makefile serve livehtml

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all: route unknown targets to Sphinx (e.g. make html).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Preview the built site (stdlib only). Use 0.0.0.0 instead of 127.0.0.1 if you
# need to reach the server from another host without SSH port forwarding.
serve: html
	python3 -m http.server 8000 --bind 127.0.0.1 --directory "$(BUILDDIR)/html"

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)"/livehtml $(SPHINXOPTS) $(O)
