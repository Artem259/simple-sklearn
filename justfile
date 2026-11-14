[windows]
set shell := ['cmd', '/c']

alias l := lint
alias t := test
alias b := build
alias c := check

_default:
    @just --list --unsorted

[group('setup')]
install:
    uv sync --locked

[group('setup')]
setup: install
    uv run pre-commit install

[group('lint')]
lint:
    uv run pre-commit run --all-files

[group('test')]
test *args:
    uv run pytest -x {{args}}

[group('test')]
test-cov *args:
    uv run pytest --cov=simple_sklearn --cov-branch --cov-report=term-missing {{args}}

[group('test')]
test-lf *args:
    uv run pytest --lf {{args}}

[group('build')]
build:
    uv run check-sdist --inject-junk
    uv build --clear
    uv run check-wheel-contents dist/*.whl
    uv run twine check --strict dist/*

[group('check')]
check: lint test build

[group('docs')]
[env("EXECUTE_NOTEBOOKS", "false")]
docs:
    uv run mkdocs serve --clean --livereload

[group('docs')]
docs-build:
    uv run mkdocs build --strict

[group('docs')]
docs-serve: docs-build
    uv run python -m http.server 8000 --bind 127.0.0.1 --directory site

act_artifact_path := justfile_directory() / "act-artifacts"

act_base := "act -P ubuntu-latest=catthehacker/ubuntu:act-latest " + \
            "--strict " + \
            "--rm " + \
            "--workflows .github/workflows/ci-cd.yml " + \
            "--matrix os:ubuntu-latest " + \
            "--artifact-server-path " + act_artifact_path + " " + \
            "--defaultbranch main"

[group('ci')]
act-push:
    {{act_base}} push \
        --env GITHUB_REF=refs/heads/main \
        --env GITHUB_BASE_REF=main

[group('ci')]
act-pr:
    {{act_base}} pull_request \
        --env GITHUB_REF=refs/pull/1/merge \
        --env GITHUB_BASE_REF=main \
        --env GITHUB_HEAD_REF=my-feature-branch
