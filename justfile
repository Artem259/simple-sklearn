[windows]
set shell := ['cmd', '/c']

alias l := lint
alias t := test
alias b := build
alias c := check

_default:
    @just --list --unsorted

# Sync the environment with locked dependencies
[group('setup')]
install:
    uv sync --locked

# Install dependencies and set up the environment
[group('setup')]
setup: install
    uv run pre-commit install

# Remove generated development artifacts
[group('setup')]
clean:
    uv run --no-project python -B scripts/clean.py

# Run pre-commit checks across all files
[group('lint')]
lint:
    uv run pre-commit run --all-files

# Run the pytest suite (stops on first failure)
[group('test')]
test:
    uv run pytest -x

# Run the pytest suite and generate a coverage report
[group('test')]
test-cov:
    uv run pytest --cov=simple_sklearn --cov-branch --cov-report=term-missing

# Rerun only the tests that failed in the previous run
[group('test')]
test-lf:
    uv run pytest --lf

# Build and validate package distributions
[group('build')]
build:
    uv run check-sdist --inject-junk
    uv build --clear
    uv run check-wheel-contents dist/*.whl
    uv run twine check --strict dist/*

# Run lint, test, and build sequentially
[group('check')]
check: lint test build

# Preview documentation with live reloading (skips notebook execution)
[group('docs')]
[env("EXECUTE_NOTEBOOKS", "false")]
docs:
    uv run mkdocs serve --clean --livereload

# Build and verify documentation (executes notebooks)
[group('docs')]
docs-build:
    uv run mkdocs build --strict

# Serve the built documentation locally on port 8000
[group('docs')]
docs-serve:
    uv run python -m http.server 8000 --bind 127.0.0.1 --directory site

act_base := "act -P ubuntu-latest=catthehacker/ubuntu:act-latest " + \
            "--strict " + \
            "--rm " + \
            "--workflows .github/workflows/ci-cd.yml " + \
            "--matrix os:ubuntu-latest " + \
            "--artifact-server-path act-artifacts " + \
            "--defaultbranch main"

# Simulate the push workflow locally with act
[group('ci')]
act-push:
    {{act_base}} push \
        --env GITHUB_REF=refs/heads/main \
        --env GITHUB_BASE_REF=main

# Simulate the pull-request workflow locally with act
[group('ci')]
act-pr:
    {{act_base}} pull_request \
        --env GITHUB_REF=refs/pull/1/merge \
        --env GITHUB_BASE_REF=main \
        --env GITHUB_HEAD_REF=my-feature-branch
