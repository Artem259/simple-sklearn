import importlib
import re
import sys

import pytest

import simple_sklearn


def test_version_is_exposed_and_valid() -> None:
    version = simple_sklearn.__version__

    assert isinstance(version, str)
    assert version != "unknown", "Version is 'unknown'. Make sure you run 'make install' before running tests."

    valid_version_pattern = re.compile(r"^\d+\.\d+")
    assert valid_version_pattern.match(version)


def test_version_fallback_on_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "simple_sklearn._version", None)

        # Reload module to force __init__.py to re-execute and hit the exception
        importlib.reload(simple_sklearn)

        assert simple_sklearn.__version__ == "unknown"

    # Reload module to restore the real __version__
    importlib.reload(simple_sklearn)
