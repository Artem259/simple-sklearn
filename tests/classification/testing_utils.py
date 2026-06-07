from typing import Any

import numpy as np
import numpy.testing as npt
from numpy.typing import NDArray


def assert_valid_classification_predictions(
    X_pred: NDArray[Any], y_pred: Any, y_train: NDArray[Any], classes: NDArray[Any]
) -> None:
    """Validates the shape, type, and content of classifier predictions."""
    assert isinstance(y_pred, np.ndarray), "Predictions must be a numpy array."
    assert len(y_pred) == len(X_pred), "Prediction length must match input length."

    unique_train_labels = np.unique(y_train)
    for p in y_pred:
        assert p in unique_train_labels, f"Prediction '{p}' not found in training labels."
        assert p in classes, f"Prediction '{p}' not found in classifier's classes_ attribute."


def assert_matches_sklearn_predictions(y_pred: NDArray[Any], sk_y_pred: NDArray[Any]) -> None:
    """Ensures custom implementation matches sklearn exactly."""
    npt.assert_array_equal(y_pred, sk_y_pred, err_msg="Predictions do not match sklearn.")
