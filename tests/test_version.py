import re

import simple_sklearn


def test_version_is_exposed_and_valid() -> None:
    version = simple_sklearn.__version__

    assert isinstance(version, str)
    assert version != "unknown", "Version is 'unknown'. Make sure you run 'make install' before running tests."

    valid_version_pattern = re.compile(r"^\d+\.\d+")
    assert valid_version_pattern.match(version)
