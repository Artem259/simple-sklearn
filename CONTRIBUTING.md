# Contributing to simple-sklearn

[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing](https://img.shields.io/pypi/types/simple-sklearn)](https://pypi.org/p/simple-sklearn)

First off, thank you for considering contributing to **simple-sklearn**!

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

* Python 3.10 or higher
* Git

### Workflow

The project uses a Makefile to simplify development workflows.

*Note for Windows users: the `make` tool is not natively installed on Windows, but manual fallback commands
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

4. **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv

    # On Linux/macOS:
    source .venv/bin/activate

    # On Windows:
    .venv\Scripts\activate
    ```

5. **Install Dependencies:**

    **Using `make` (Linux/macOS):**

    This command installs the package in editable mode along with all necessary dev dependencies,
    and sets up the `pre-commit` hooks.

    ```bash
    make setup
    ```

    **Manual commands (Windows):**

    ```bash
    python -m pip install --upgrade pip
    python -m pip install -e ".[docs,quality,build,test,dev]"
    pre-commit install
    ```

6. **Run checks and tests:**

    The project uses `ruff` for formatting/linting, `mypy` for static type checking, and `pytest` for testing.

    ```bash
    make lint        # Runs pre-commit checks across all files
    make test        # Runs the pytest suite
    make test-cov    # Runs the pytest suite and generates a coverage report
    make check       # Runs linting, build verification (sdist/wheel), and tests sequentially
    ```

    You can also run `pytest` and `pre-commit run --all-files` manually.

7. **Local CI/CD Simulation (Optional):**

    If you have [act](https://nektosact.com/) installed locally, you can simulate the GitHub Actions pipeline
    before opening a PR:

    ```bash
    make act-pr      # Simulates the pull-request workflow locally
    ```

8. **Build documentation locally:**

    The project uses `mkdocs` with the `mkdocs-jupyter` plugin for demo Jupyter notebooks.

    ```bash
    make docs        # Preview documentation with live-reloading (no notebook execution)
    make docs-build  # Build and verify documentation
    ```

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
* [ ] `make check` and `make docs-build` (or the equivalent manual commands) pass locally without errors.
* [ ] I have added or updated tests for my changes, and my changes do not decrease overall test coverage.
* [ ] I have successfully run scikit-learn estimator checks against any new estimators.
* [ ] I have updated the docstrings and documentation if my changes affect the API.
* [ ] My PR title follows the Conventional Commits specification.
