import typing
from typing import Any, get_args, get_origin

import numpy as np
import pytest
from numpy.typing import NDArray


def _unwrap_alias(type_hint: Any) -> Any:
    """Unwraps PEP 695 TypeAliasType down to its underlying value."""
    TypeAliasType = getattr(typing, "TypeAliasType", ())
    while isinstance(type_hint, TypeAliasType):
        type_hint = type_hint.__value__
    return type_hint


def assert_matches_type(obj: Any, type_hint: Any) -> None:
    """Recursively asserts that an object matches a nested type hint (e.g., list[list[int]])."""
    if type_hint is Any:
        return
    type_hint = _unwrap_alias(type_hint)

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is None:
        assert isinstance(obj, type_hint), (
            f"Expected {getattr(type_hint, '__name__', str(type_hint))}, got {type(obj).__name__}"
        )
        return
    origin = get_origin(_unwrap_alias(origin)) or _unwrap_alias(origin)

    assert isinstance(obj, origin), f"Expected {getattr(origin, '__name__', str(origin))}, got {type(obj).__name__}"

    if origin is np.ndarray:
        # If the array type is unparameterized (e.g., just `NDArray` or `np.ndarray`)
        if not args:
            return

        dtype_hint = args[1] if len(args) == 2 else args[0]

        origin_dtype = get_origin(dtype_hint) or getattr(dtype_hint, "__origin__", None)
        if origin_dtype is np.dtype:
            dtype_args = get_args(dtype_hint)
            expected_dtype = dtype_args[0] if dtype_args else Any
        else:
            expected_dtype = dtype_hint

        if expected_dtype is not Any:
            assert np.issubdtype(obj.dtype, expected_dtype), (
                f"Expected array of dtype {getattr(expected_dtype, '__name__', str(expected_dtype))}, got {obj.dtype}"
            )
        return

    if args and not isinstance(obj, (str, bytes)):
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


def assert_parameter_validation_exceptions(
    estimator: Any,
    X: NDArray[Any],
    y: NDArray[Any] | None,
    expected_error: type[Exception],
    match_text: str,
    sk_estimator: Any = None,
) -> None:
    with pytest.raises(expected_error, match=match_text):
        estimator.fit(X, y)

    if sk_estimator is not None:
        with pytest.raises(expected_error, match=match_text):
            sk_estimator.fit(X, y)
