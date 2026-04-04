.PHONY: install setup lint build test check docs docs-build act-push act-pr

.DEFAULT_GOAL := setup

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[docs,quality,test,dev]"

setup: install
	pre-commit install

lint:
	pre-commit run --all-files

build:
	python -m check_sdist --inject-junk
	python -m build
	python -m twine check --strict dist/*

test:
	pytest

check: lint build test

docs: export EXECUTE_NOTEBOOKS=false
docs:
	mkdocs serve --clean --livereload

docs-build:
	mkdocs build --strict
	python -m http.server 8000 --bind 127.0.0.1 --directory site

ACT_BASE := act -P ubuntu-latest=catthehacker/ubuntu:act-latest \
                 --matrix os:ubuntu-latest \
                 --artifact-server-path ./act-artifacts \
                 --defaultbranch main \
                 --env GITHUB_REF=refs/heads/main \
                 --env GITHUB_BASE_REF=main

act-push:
	$(ACT_BASE) push

act-pr:
	$(ACT_BASE) pull_request
