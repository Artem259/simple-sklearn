.PHONY: install setup lint build test check test-ci

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

test-ci:
	act -P ubuntu-latest=catthehacker/ubuntu:act-latest --matrix os:ubuntu-latest --artifact-server-path ./act-artifacts
