from typing import Any, get_args, get_origin

import numpy as np


def assert_matches_type(obj: Any, type_hint: Any) -> None:
    """Recursively asserts that an object matches a nested type hint (e.g., list[list[int]])."""
    if type_hint is Any:
        return

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is None:
        assert isinstance(obj, type_hint), (
            f"Expected {getattr(type_hint, '__name__', str(type_hint))}, got {type(obj).__name__}"
        )
        return

    assert isinstance(obj, origin), f"Expected {origin.__name__}, got {type(obj).__name__}"

    if origin is np.ndarray:
        # NDArray type hints are typically formatted as ndarray[Shape, DType]
        if len(args) == 2:
            dtype_hint = args[1]  # e.g., numpy.dtype[numpy.float64]
            dtype_args = get_args(dtype_hint)  # e.g., (numpy.float64,)
            if dtype_args and dtype_args[0] is not Any:
                expected_dtype = dtype_args[0]
                assert np.issubdtype(obj.dtype, expected_dtype), (
                    f"Expected array of dtype {getattr(expected_dtype, '__name__', str(expected_dtype))}, "
                    f"got {obj.dtype}"
                )
        return

    if args and hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        expected_inner_type = args[0]
        for i, item in enumerate(obj):
            try:
                assert_matches_type(item, expected_inner_type)
            except AssertionError as e:
                raise AssertionError(f"At index [{i}] -> {e}") from None


def assert_attributes_match_types(estimator: Any, type_map: dict[str, Any]) -> None:
    for attr_name, expected_type in type_map.items():
        assert hasattr(estimator, attr_name), (
            f"Expected attribute '{attr_name}' is missing from {type(estimator).__name__}"
        )
        attr_value = getattr(estimator, attr_name)

        try:
            assert_matches_type(attr_value, expected_type)
        except AssertionError as e:
            raise AssertionError(f"Type mismatch for attribute '{attr_name}': {e}") from e
