from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
from numpy.typing import NDArray

from simple_sklearn.clustering import KMedoids
from tests.testing_utils import (
    assert_attributes_match_types,
    assert_parameter_validation_exceptions,
)


@pytest.mark.parametrize("n_clusters", list(range(1, 11)))
@pytest.mark.parametrize("max_iter", [1, 2, 5, 10, 300])
@pytest.mark.parametrize("_init", ["random", None, "_duplicate_centers"])
def test_kmedoids_fit(dataset_2f: NDArray[np.int64], n_clusters: int, max_iter: int, _init: Any) -> None:
    X = dataset_2f
    n_samples, n_features = X.shape

    if n_clusters > n_samples:
        pytest.skip(f"n_clusters ({n_clusters}) cannot exceed n_samples ({n_samples})")

    if _init == "_duplicate_centers":
        # Initialize all clusters to the exact same point (the first sample of X)
        # to test empty clusters handling
        init = [X[0].tolist()] * n_clusters
    elif _init is None:
        init = X[:n_clusters].tolist()
    else:
        init = _init

    expected_attributes = {
        "cluster_centers_": NDArray[np.float64],
        "labels_": NDArray[np.int64],
        "n_iter_": int,
        "inertia_": float,
        "random_state_": np.random.RandomState,
        "distance_matrix_": NDArray[np.float64],
        "cluster_center_indices_": NDArray[np.int64],
    }

    clusterer = KMedoids(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=42)
    clusterer.fit(X)

    # Validate expected learned attributes
    assert_attributes_match_types(clusterer, expected_attributes)

    # Validate cluster_centers_ attribute
    assert clusterer.cluster_centers_.shape == (n_clusters, n_features)
    # Cluster centers must be finite numerical values
    assert np.all(np.isfinite(clusterer.cluster_centers_))

    # Validate labels_ attribute
    assert clusterer.labels_.shape == (n_samples,)
    # Labels should be valid cluster IDs (between 0 and n_clusters - 1)
    assert np.all(clusterer.labels_ >= 0)
    assert np.all(clusterer.labels_ < n_clusters)

    # Validate n_iter_ attribute
    # The algorithm must perform at least 1 iteration, up to a maximum of max_iter
    assert 1 <= clusterer.n_iter_ <= max_iter

    # Validate inertia_ attribute
    # The sum of distances must be non-negative
    assert clusterer.inertia_ >= 0.0

    # Validate random_state_ attribute
    pass

    # Validate distance_matrix_ attribute
    assert clusterer.distance_matrix_.shape == (n_samples, n_samples)
    assert np.all(clusterer.distance_matrix_ >= 0.0)
    # The distance matrix must be symmetric and have a zero diagonal
    npt.assert_allclose(np.diag(clusterer.distance_matrix_), 0.0, atol=1e-8)
    npt.assert_allclose(clusterer.distance_matrix_, clusterer.distance_matrix_.T, rtol=1e-6, atol=1e-8)

    # Validate cluster_center_indices_ attribute
    assert clusterer.cluster_center_indices_.shape == (n_clusters,)
    assert np.all(clusterer.cluster_center_indices_ >= 0)
    assert np.all(clusterer.cluster_center_indices_ < n_samples)
    # Ensure that the medoids exactly match the data points at these indices
    npt.assert_array_equal(clusterer.cluster_centers_, X[clusterer.cluster_center_indices_])


@pytest.mark.parametrize(
    ("params", "expected_error", "match_text"),
    [
        # n_clusters checks
        ({"n_clusters": 0.0}, ValueError, r"of KMedoids must be an int in the range \[1, inf\)\. Got 0\.0"),
        ({"n_clusters": 0}, ValueError, r"of KMedoids must be an int in the range \[1, inf\)\. Got 0"),
        ({"n_clusters": -1}, ValueError, r"of KMedoids must be an int in the range \[1, inf\)\. Got -1"),
        ({"n_clusters": "1e-9"}, ValueError, r"of KMedoids must be an int in the range \[1, inf\)\. Got 1e-9"),
        ({"n_clusters": 11}, ValueError, r"n_samples=10 should be >= n_clusters=11"),
        # init checks
        (
            {"init": "invalid_init"},
            ValueError,
            r"of KMedoids must be array-like or a str among \{'random'\}\. Got 'invalid_init'",
        ),
        (
            {"init": 123},
            ValueError,
            r"of KMedoids must be array-like or a str among \{'random'\}\. Got '123'",
        ),
        (
            {"init": None},
            ValueError,
            r"of KMedoids must be array-like or a str among \{'random'\}\. Got 'None'",
        ),
        (
            {"n_clusters": 5, "init": [[1, 2]] * 6},
            ValueError,
            r"\(6, 2\) does not match the expected shape \(n_clusters, n_features\): \(5, 2\)",
        ),
        (
            {"n_clusters": 5, "init": [[1, 2, 3]] * 5},
            ValueError,
            r"\(5, 3\) does not match the expected shape \(n_clusters, n_features\): \(5, 2\)",
        ),
        (
            {"n_clusters": 5, "init": [["a", "a"]] * 5},
            ValueError,
            r"could not convert string to float: 'a'",
        ),
        # max_iter checks
        ({"max_iter": 0.0}, ValueError, r"of KMedoids must be an int in the range \[1, inf\)\. Got 0\.0"),
        ({"max_iter": 0}, ValueError, r"of KMedoids must be an int in the range \[1, inf\)\. Got 0"),
        ({"max_iter": -1}, ValueError, r"of KMedoids must be an int in the range \[1, inf\)\. Got -1"),
        ({"max_iter": "1e-9"}, ValueError, r"of KMedoids must be an int in the range \[1, inf\)\. Got 1e-9"),
    ],
)
def test_kmedoids_parameter_validation_exceptions(
    dataset_2f: NDArray[np.int64],
    params: dict[str, Any],
    expected_error: type[Exception],
    match_text: str,
) -> None:
    X = dataset_2f

    clusterer = KMedoids(**params)

    assert_parameter_validation_exceptions(clusterer, X, None, expected_error, match_text, None)
