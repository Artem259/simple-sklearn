import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.cluster import adjusted_rand_score


def assert_matches_sklearn_cluster_labels(labels: NDArray[np.int64], sk_labels: NDArray[np.int64]) -> None:
    """Ensures custom implementation matches sklearn."""
    ars = adjusted_rand_score(labels, sk_labels)
    assert ars == 1.0
