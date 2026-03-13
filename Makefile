.PHONY: install setup lint test check build test-ci

.DEFAULT_GOAL := setup

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[jupyter,quality,test,dev]"

setup: install
	pre-commit install

lint:
	pre-commit run --all-files

test:
	pytest

check: lint test

build:
	python -m build
	python -m twine check --strict dist/*

test-ci:
	act -P ubuntu-latest=catthehacker/ubuntu:act-latest --matrix os:ubuntu-latest --artifact-server-path ./act-artifacts
