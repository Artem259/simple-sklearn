import numpy as np
import numpy.testing as npt
import pytest
import sklearn
from numpy.typing import NDArray

from simple_sklearn.classification import KNeighborsClassifier
from tests.classification.testing_utils import (
    assert_matches_sklearn_predictions,
    assert_valid_classification_predictions,
)
from tests.testing_utils import (
    assert_attributes_match_types,
)


@pytest.mark.parametrize("n_neighbors", list(range(1, 11)))
@pytest.mark.parametrize(
    ("weights", "match_sklearn"),
    [
        ("uniform", True),
        ("distance", True),
        ("distance_squared", False),
    ],
)
def test_kneighbors_matches_sklearn_on_simple_data(
    dataset_4f: tuple[NDArray[np.int_], NDArray[np.str_]], n_neighbors: int, weights: str, match_sklearn: bool
) -> None:
    X, y = dataset_4f
    n_samples, n_features = X.shape
    unique_classes = set(y)
    n_classes = len(unique_classes)
    encoded_classes = set(range(n_classes))

    if n_neighbors > n_samples:
        pytest.skip(f"n_neighbors ({n_neighbors}) cannot exceed n_samples ({n_samples})")

    X_pred = np.array(
        [
            [2, 1, 0, 1],
            [0, 0, 0, 0],
            [3.5, 1, 1.2, 1],
            [-0.25, 2, -1, 1.7],
        ]
    )

    expected_attributes = {
        "classes_": NDArray[np.str_],
        "fitted_x_": NDArray[np.int_],
        "fitted_y_": NDArray[np.int_],
    }

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    clf.fit(X, y)

    # Validate expected learned attributes
    assert_attributes_match_types(clf, expected_attributes)

    # Validate classes_ attribute
    assert clf.classes_.shape[0] == n_classes
    assert set(clf.classes_) == unique_classes

    # Validate fitted_x_ attribute
    assert clf.fitted_x_.shape == (n_samples, n_features)
    npt.assert_array_equal(clf.fitted_x_, X)

    # Validate fitted_y_ attribute
    assert clf.fitted_y_.shape == (n_samples,)
    assert set(clf.fitted_y_) == encoded_classes

    # Check predictions
    y_pred = clf.predict(X_pred)
    assert_valid_classification_predictions(X_pred, y_pred, y, clf.classes_)

    # Compare against sklearn's classifier on the same data
    if match_sklearn:
        sk = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        sk.fit(X, y)

        # Compare predictions
        sk_y_pred = sk.predict(X_pred)
        assert_matches_sklearn_predictions(y_pred, sk_y_pred)

        # Compare kneighbors output distances
        dist, _ = clf.kneighbors(X_pred)
        sk_dist, _ = sk.kneighbors(X_pred)
        npt.assert_allclose(dist, sk_dist, rtol=1e-6, atol=1e-8)
