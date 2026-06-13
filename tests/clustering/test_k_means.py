from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
import sklearn
from numpy.typing import NDArray

from simple_sklearn.clustering import KMeans
from tests.clustering.testing_utils import (
    assert_matches_sklearn_cluster_labels,
)
from tests.testing_utils import (
    assert_attributes_match_types,
    assert_parameter_validation_exceptions,
)


@pytest.mark.parametrize("n_clusters", list(range(1, 11)))
@pytest.mark.parametrize("max_iter", [1, 2, 5, 10, 300])
@pytest.mark.parametrize(
    ("_init", "match_sklearn"),
    [
        ("random", False),
        (None, True),
    ],
)
def test_kmeans_matches_sklearn_on_simple_data(
    dataset_2f: NDArray[np.int64], n_clusters: int, max_iter: int, _init: Any, match_sklearn: bool
) -> None:
    X = dataset_2f
    n_samples, n_features = X.shape

    if n_clusters > n_samples:
        pytest.skip(f"n_clusters ({n_clusters}) cannot exceed n_samples ({n_samples})")

    init = X[:n_clusters].tolist() if _init is None else _init

    expected_attributes = {
        "cluster_centers_": NDArray[np.float64],
        "labels_": NDArray[np.int64],
        "n_iter_": int,
        "inertia_": float,
        "random_state_": np.random.RandomState,
    }

    clusterer = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, atol=0.0, random_state=42)
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
    # The sum of squared distances must be non-negative
    assert clusterer.inertia_ >= 0.0

    # Validate random_state_ attribute
    pass

    # Compare against sklearn's clusterer on the same data
    if match_sklearn:
        # Enforce classic 'lloyd' algorithm and a single initialization run (n_init=1)
        # to guarantee 1:1 behavioral parity with custom implementation
        sk = sklearn.cluster.KMeans(
            n_clusters=n_clusters, init=init, max_iter=max_iter, tol=0.0, n_init=1, algorithm="lloyd", random_state=42
        )
        sk.fit(X)

        # Compare cluster labels
        assert_matches_sklearn_cluster_labels(clusterer.labels_, sk.labels_)

        # Compare cluster_centers_ and inertia_ attributes
        npt.assert_allclose(clusterer.cluster_centers_, sk.cluster_centers_, rtol=1e-6, atol=1e-8)
        npt.assert_allclose(clusterer.inertia_, sk.inertia_, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    ("params", "expected_error", "match_text", "match_sklearn"),
    [
        # n_clusters checks
        ({"n_clusters": 0.0}, ValueError, r"of KMeans must be an int in the range \[1, inf\)\. Got 0\.0", True),
        ({"n_clusters": 0}, ValueError, r"of KMeans must be an int in the range \[1, inf\)\. Got 0", True),
        ({"n_clusters": -1}, ValueError, r"of KMeans must be an int in the range \[1, inf\)\. Got -1", True),
        ({"n_clusters": "1e-9"}, ValueError, r"of KMeans must be an int in the range \[1, inf\)\. Got 1e-9", False),
        ({"n_clusters": 11}, ValueError, r"n_samples=10 should be >= n_clusters=11", True),
        # init checks
        (
            {"init": "invalid_init"},
            ValueError,
            r"of KMeans must be array-like or a str among \{'random'\}\. Got 'invalid_init'",
            False,
        ),
        (
            {"init": 123},
            ValueError,
            r"of KMeans must be array-like or a str among \{'random'\}\. Got '123'",
            False,
        ),
        (
            {"init": None},
            ValueError,
            r"of KMeans must be array-like or a str among \{'random'\}\. Got 'None'",
            False,
        ),
        (
            {"n_clusters": 5, "init": [[1, 2]] * 6},
            ValueError,
            r"\(6, 2\) does not match the expected shape \(n_clusters, n_features\): \(5, 2\)",
            False,
        ),
        (
            {"n_clusters": 5, "init": [[1, 2, 3]] * 5},
            ValueError,
            r"\(5, 3\) does not match the expected shape \(n_clusters, n_features\): \(5, 2\)",
            False,
        ),
        (
            {"n_clusters": 5, "init": [["a", "a"]] * 5},
            ValueError,
            r"could not convert string to float: 'a'",
            False,
        ),
        # max_iter checks
        ({"max_iter": 0.0}, ValueError, r"of KMeans must be an int in the range \[1, inf\)\. Got 0\.0", True),
        ({"max_iter": 0}, ValueError, r"of KMeans must be an int in the range \[1, inf\)\. Got 0", True),
        ({"max_iter": -1}, ValueError, r"of KMeans must be an int in the range \[1, inf\)\. Got -1", True),
        ({"max_iter": "1e-9"}, ValueError, r"of KMeans must be an int in the range \[1, inf\)\. Got 1e-9", False),
        # atol checks
        ({"atol": -0.1}, ValueError, r"must be a float in the range \[0\.0, inf\)\. Got -0\.1", False),
        ({"atol": "1e-9"}, ValueError, r"must be a float in the range \[0\.0, inf\)\. Got 1e-9", False),
    ],
)
def test_kmeans_parameter_validation_exceptions(
    dataset_2f: NDArray[np.int64],
    params: dict[str, Any],
    expected_error: type[Exception],
    match_text: str,
    match_sklearn: bool,
) -> None:
    X = dataset_2f

    clusterer = KMeans(**params)
    sk = sklearn.cluster.KMeans(**params) if match_sklearn else None

    assert_parameter_validation_exceptions(clusterer, X, None, expected_error, match_text, sk)
