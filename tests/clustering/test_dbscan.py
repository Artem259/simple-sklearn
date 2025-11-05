import numpy as np
from sklearn.cluster import DBSCAN as SKDBSCAN
from src.simple_sklearn.clustering.DBSCAN import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.utils.estimator_checks import estimator_checks_generator


def test_dbscan_matches_sklearn_on_simple_data():
    # two compact clusters + one noise point
    X = np.array([
        [0.0, 0.0],
        [0.1, -0.1],
        [0.0, 0.2],
        [5.0, 5.0],
        [5.1, 4.9],
        [100.0, 100.0],  # isolated noise
    ])
    eps = 0.5
    min_samples = 2

    custom = DBSCAN(eps=eps, min_samples=min_samples)
    custom.fit(X)

    sk = SKDBSCAN(eps=eps, min_samples=min_samples)
    sk.fit(X)

    ars = adjusted_rand_score(custom.labels_, sk.labels_)
    assert ars == 1.0

    assert hasattr(custom, "core_sample_indices_")
    assert isinstance(custom.core_sample_indices_, (list, np.ndarray))


def test_dbscan_passes_sklearn_checks():
    clusterer = DBSCAN()
    for (estimator, check) in estimator_checks_generator(clusterer):
        check(estimator)
