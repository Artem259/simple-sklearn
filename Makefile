.PHONY: install setup lint build test test-cov test-lf check docs docs-build act-push act-pr

.DEFAULT_GOAL := setup

install:
	uv sync --locked

setup: install
	uv run pre-commit install

lint:
	uv run pre-commit run --all-files

build:
	uv run check-sdist --inject-junk
	uv build --clear
	uv run check-wheel-contents dist/*.whl
	uv run twine check --strict dist/*

test:
	uv run pytest -x

test-cov:
	uv run pytest --cov=simple_sklearn --cov-branch --cov-report=term-missing

test-lf:
	uv run pytest --lf

check: lint build test

docs: export EXECUTE_NOTEBOOKS=false
docs:
	uv run mkdocs serve --clean --livereload

docs-build:
	uv run mkdocs build --strict
	uv run python -m http.server 8000 --bind 127.0.0.1 --directory site

ACT_BASE := act -P ubuntu-latest=catthehacker/ubuntu:act-latest \
                 --strict \
                 --rm \
                 --workflows .github/workflows/ci-cd.yml \
                 --matrix os:ubuntu-latest \
                 --artifact-server-path $(CURDIR)/act-artifacts \
                 --defaultbranch main

act-push:
	$(ACT_BASE) push \
		--env GITHUB_REF=refs/heads/main \
		--env GITHUB_BASE_REF=main

act-pr:
	$(ACT_BASE) pull_request \
		--env GITHUB_REF=refs/pull/1/merge \
		--env GITHUB_BASE_REF=main \
		--env GITHUB_HEAD_REF=my-feature-branch
