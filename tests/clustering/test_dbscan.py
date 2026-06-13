from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
import sklearn
from numpy.typing import NDArray

from simple_sklearn.clustering import DBSCAN
from tests.clustering.testing_utils import (
    assert_matches_sklearn_cluster_labels,
)
from tests.testing_utils import (
    assert_attributes_match_types,
    assert_parameter_validation_exceptions,
)


@pytest.mark.parametrize("eps", [0.1, 1, 3, 5, 10])
@pytest.mark.parametrize("min_samples", [*list(range(1, 11)), 99])
@pytest.mark.parametrize("match_sklearn", [True])
def test_dbscan_matches_sklearn_on_simple_data(
    dataset_2f: NDArray[np.int64], eps: float, min_samples: int, match_sklearn: bool
) -> None:
    X = dataset_2f
    n_samples = X.shape[0]

    expected_attributes = {
        "labels_": NDArray[np.int64],
        "distance_matrix_": NDArray[np.float64],
        "neighbors_": list[list[int]],
        "core_sample_indices_": NDArray[np.int64],
        "_core_sample_mask": NDArray[np.bool_],
    }

    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    clusterer.fit(X)

    # Validate expected learned attributes
    assert_attributes_match_types(clusterer, expected_attributes)

    # Validate labels_ attribute
    assert clusterer.labels_.shape == (n_samples,)
    # Labels should be valid cluster IDs (>= 0) or -1 for noise
    assert np.all(clusterer.labels_ >= -1)

    # Validate distance_matrix_ attribute
    assert clusterer.distance_matrix_.shape == (n_samples, n_samples)
    assert np.all(clusterer.distance_matrix_ >= 0.0)
    # The distance matrix must be symmetric and have a zero diagonal
    npt.assert_allclose(np.diag(clusterer.distance_matrix_), 0.0, atol=1e-8)
    npt.assert_allclose(clusterer.distance_matrix_, clusterer.distance_matrix_.T, rtol=1e-6, atol=1e-8)

    # Validate neighbors_ attribute
    assert len(clusterer.neighbors_) == n_samples
    for i, neighbors in enumerate(clusterer.neighbors_):
        # The algorithm excludes the point itself from its neighbor list
        assert i not in neighbors
        if neighbors:
            assert min(neighbors) >= 0
            assert max(neighbors) < n_samples

    # Validate _core_sample_mask attribute
    assert clusterer._core_sample_mask.shape == (n_samples,)

    # Validate core_sample_indices_ attribute
    assert clusterer.core_sample_indices_.ndim == 1
    # Ensure the indices explicitly match where the boolean mask is True
    expected_core_indices = np.where(clusterer._core_sample_mask)[0]
    npt.assert_array_equal(clusterer.core_sample_indices_, expected_core_indices)

    # Compare against sklearn's clusterer on the same data
    if match_sklearn:
        sk = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)
        sk.fit(X)

        # Compare cluster labels
        assert_matches_sklearn_cluster_labels(clusterer.labels_, sk.labels_)


@pytest.mark.parametrize(
    ("params", "expected_error", "match_text", "match_sklearn"),
    [
        # eps checks
        ({"eps": 0.0}, ValueError, r"must be a float in the range \(0\.0, inf\)\. Got 0\.0", True),
        ({"eps": -0.1}, ValueError, r"must be a float in the range \(0\.0, inf\)\. Got -0\.1", True),
        ({"eps": "1e-9"}, ValueError, r"must be a float in the range \(0\.0, inf\)\. Got 1e-9", False),
        # min_samples checks
        ({"min_samples": 0.0}, ValueError, r"must be an int in the range \[1, inf\)\. Got 0\.0", True),
        ({"min_samples": 0}, ValueError, r"must be an int in the range \[1, inf\)\. Got 0", True),
        ({"min_samples": -1}, ValueError, r"must be an int in the range \[1, inf\)\. Got -1", True),
        ({"min_samples": "1e-9"}, ValueError, r"must be an int in the range \[1, inf\)\. Got 1e-9", False),
    ],
)
def test_dbscan_parameter_validation_exceptions(
    dataset_2f: NDArray[np.int64],
    params: dict[str, Any],
    expected_error: type[Exception],
    match_text: str,
    match_sklearn: bool,
) -> None:
    X = dataset_2f

    clusterer = DBSCAN(**params)
    sk = sklearn.cluster.DBSCAN(**params) if match_sklearn else None

    assert_parameter_validation_exceptions(clusterer, X, None, expected_error, match_text, sk)
