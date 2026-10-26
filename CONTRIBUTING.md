# Contributing to simple-sklearn

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)][uv-url]
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)][ruff-url]
[![Typing](https://img.shields.io/pypi/types/simple-sklearn)][pypi-url]

Before starting work on a major feature or architectural change, please open an issue to discuss it.
This ensures your approach aligns with the project's roadmap and prevents wasted effort.

---

## Conventional Commits

This repository uses [Release Please](https://github.com/googleapis/release-please) to automate
changelog generation and GitHub releases.

Because of this, **all Pull Request titles must follow the
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.**

* Example feature: `feat: implement KDTree for neighbors search`
* Example bug fix: `fix(dbscan): resolve zero-division error`
* Example docs: `docs: update roadmap in README`

If your PR does not follow this structure, the automated CI/CD pipeline will fail to interpret
the changes for semantic versioning.

---

## Development Setup

### Prerequisites

* [uv](https://docs.astral.sh/uv/) (automatically installs Python 3.10, creates `.venv`, and manages dependencies)
* Git

### Workflow

The project uses `uv` for dependency and environment management, alongside a `Makefile` to simplify
common development tasks.

*Note for Windows users: the `make` tool is not natively installed on Windows, so direct `uv` fallback commands
are provided below.*

1. **Fork and clone the repository:**

    Click the "Fork" button at the top right of the repository page, then clone your fork locally:

    ```bash
    git clone https://github.com/<YOUR-USERNAME>/simple-sklearn.git
    cd simple-sklearn
    ```

2. **Add the upstream remote:**

    This allows you to keep your fork synced with the main repository:

    ```bash
    git remote add upstream https://github.com/Artem259/simple-sklearn.git
    ```

3. **Create a feature branch:**

    Always create a new branch for your work rather than committing to `main`:

    ```bash
    git checkout -b feat/my-new-feature
    ```

4. **Install dependencies and set up the environment:**

    `uv` automatically creates a `.venv` virtual environment in the project root and installs the
    package in editable mode along with all locked development dependencies.

    **Using `make` (Linux/macOS):**

    ```bash
    make setup
    ```

    **Manual commands (Windows / without `make`):**

    ```bash
    uv sync --locked
    uv run pre-commit install
    ```

5. **Run checks and tests:**

    The project uses `ruff` for formatting/linting, `mypy` for static type checking, and `pytest` for testing.

    ```bash
    make lint        # Runs pre-commit checks across all files
    make test        # Runs the pytest suite (stops on first failure)
    make test-cov    # Runs the pytest suite and generates a coverage report
    make check       # Runs linting, build verification (sdist/wheel), and tests sequentially
    ```

    You can also run `uv run pytest -x` and `uv run pre-commit run --all-files` manually.

6. **Local CI/CD Simulation (Optional):**

    If you have [act](https://nektosact.com/) installed locally, you can simulate the GitHub Actions pipeline
    before opening a PR:

    ```bash
    make act-pr      # Simulates the pull-request workflow locally
    ```

7. **Build documentation locally:**

    The project uses `mkdocs` with the `mkdocs-jupyter` plugin for demo Jupyter notebooks.

    ```bash
    make docs        # Preview documentation with live-reloading (no notebook execution)
    make docs-build  # Build and verify documentation strictly, then serve on port 8000
    ```

    You can also run `uv run mkdocs build --strict` manually.

---

## Coding Standards & Guidelines

To maintain the quality of code and architecture of simple-sklearn, please adhere to the following:

* **Scikit-learn Compatibility:** Any new estimator must inherit from `BaseEstimator`, the appropriate mixin
(`ClassifierMixin`/`ClusterMixin`), and successfully pass `sklearn.utils.estimator_checks`.
* **Type Hinting:** The project utilizes strict `mypy` typing. Ensure all function signatures and
class attributes are appropriately type-hinted.
* **Docstrings:** Use **Google-style docstrings**. This is strictly required so that `mkdocstrings`
can correctly generate the API reference documentation.

---

## Submitting Your Pull Request

Before submitting your PR, please ensure you have completed the following checklist:

* [ ] I have fetched the latest changes from `upstream/main` and rebased my branch if necessary.
* [ ] My PR is narrowly scoped to address a single feature or bug fix.
* [ ] `make check` and `make docs-build` (or the equivalent `uv run` commands) pass locally without errors.
* [ ] I have added or updated tests for my changes, and my changes do not decrease overall test coverage.
* [ ] I have successfully run scikit-learn estimator checks against any new estimators.
* [ ] I have updated the docstrings and documentation if my changes affect the API.
* [ ] My PR title follows the Conventional Commits specification.

[uv-url]: https://github.com/astral-sh/uv
[ruff-url]: https://github.com/astral-sh/ruff
[pypi-url]: https://pypi.org/p/simple-sklearn
