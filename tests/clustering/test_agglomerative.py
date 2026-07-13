from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
import sklearn
from numpy.typing import NDArray

from simple_sklearn.clustering import AgglomerativeClustering
from tests.testing_utils import (
    assert_attributes_match_types,
    assert_parameter_validation_exceptions,
)


@pytest.mark.parametrize("n_clusters", list(range(1, 11)))
@pytest.mark.parametrize("linkage", ["ward", "complete", "average", "single"])
@pytest.mark.parametrize("match_sklearn", [True])
def test_agglomerative_matches_sklearn_on_simple_data(
    dataset_2f: NDArray[np.int64], n_clusters: int, linkage: str, match_sklearn: bool
) -> None:
    X = dataset_2f
    n_samples = X.shape[0]

    if n_clusters > n_samples:
        pytest.skip(f"n_clusters ({n_clusters}) cannot exceed n_samples ({n_samples})")

    expected_attributes = {
        "labels_": NDArray[np.int64],
        "children_": NDArray[np.int64],
        "distances_": NDArray[np.float64],
    }

    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clusterer.fit(X)

    # Validate expected learned attributes
    assert_attributes_match_types(clusterer, expected_attributes)

    # Validate labels_ attribute
    assert clusterer.labels_.shape == (n_samples,)
    # Labels should be valid cluster IDs (between 0 and n_clusters - 1)
    assert np.all(clusterer.labels_ >= 0)
    assert np.all(clusterer.labels_ < n_clusters)

    # Validate children_ attribute
    # A hierarchical tree on N samples takes exactly N-1 merges to form a single root
    assert clusterer.children_.shape == (n_samples - 1, 2)
    # In agglomerative clustering, the node formed at iteration `i` is assigned
    # the index `n_samples + i`. Thus, its children must have indices strictly
    # less than `n_samples + i` (they must have been formed in a prior step).
    for i, (child1, child2) in enumerate(clusterer.children_):
        assert child1 < n_samples + i
        assert child2 < n_samples + i

    # Validate distances_ attribute
    assert clusterer.distances_.shape == (n_samples - 1,)
    # Distance metrics must always be non-negative
    assert np.all(clusterer.distances_ >= 0.0)

    # Compare against sklearn's clusterer on the same data
    if match_sklearn:
        sk = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, compute_distances=True)
        sk.fit(X)

        # Compare distances_ attributes
        # Rely on distances rather than children_ for comparison because tie-breaking logic for
        # equal distances can cause topological ordering differences between implementations.
        npt.assert_allclose(clusterer.distances_, sk.distances_, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    ("params", "expected_error", "match_text", "match_sklearn"),
    [
        # n_clusters checks
        ({"n_clusters": 0.0}, ValueError, r"must be an int in the range \[1, inf\)\. Got 0\.0", False),
        ({"n_clusters": 0}, ValueError, r"must be an int in the range \[1, inf\)\. Got 0", False),
        ({"n_clusters": -1}, ValueError, r"must be an int in the range \[1, inf\)\. Got -1", False),
        ({"n_clusters": "1e-9"}, ValueError, r"must be an int in the range \[1, inf\)\. Got 1e-9", False),
        ({"n_clusters": 11}, ValueError, r"11 clusters were given for a tree with 10 leaves", True),
        # linkage checks
        (
            {"linkage": "invalid_linkage"},
            ValueError,
            r"must be a str among \{'single', 'complete', 'average', 'ward'\}\. Got 'invalid_linkage'",
            False,
        ),
        (
            {"linkage": 123},
            ValueError,
            r"must be a str among \{'single', 'complete', 'average', 'ward'\}\. Got '123'",
            False,
        ),
        (
            {"linkage": None},
            ValueError,
            r"must be a str among \{'single', 'complete', 'average', 'ward'\}\. Got 'None'",
            False,
        ),
    ],
)
def test_agglomerative_parameter_validation_exceptions(
    dataset_2f: NDArray[np.int64],
    params: dict[str, Any],
    expected_error: type[Exception],
    match_text: str,
    match_sklearn: bool,
) -> None:
    X = dataset_2f

    clusterer = AgglomerativeClustering(**params)
    sk = sklearn.cluster.AgglomerativeClustering(**params) if match_sklearn else None

    assert_parameter_validation_exceptions(clusterer, X, None, expected_error, match_text, sk)
