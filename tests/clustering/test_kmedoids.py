import numpy as np
from sklearn.utils.estimator_checks import estimator_checks_generator

from simple_sklearn.clustering import KMedoids


def test_kmedoids_basic_properties() -> None:
    # small dataset with two clusters
    rng = np.random.RandomState(0)
    X = np.vstack(
        [
            rng.normal(loc=[0, 0], scale=0.1, size=(5, 2)),
            rng.normal(loc=[5, 5], scale=0.1, size=(5, 2)),
        ]
    )
    n_clusters = 2
    init = np.array([[0, 0], [5, 5]])

    km = KMedoids(n_clusters=n_clusters, init=init, random_state=0)
    km.fit(X)

    # labels length equals samples
    assert len(km.labels_) == X.shape[0]
    # cluster_center_indices_ valid integers within [0, n_samples)
    assert getattr(km, "cluster_center_indices_", None) is not None
    for idx in km.cluster_center_indices_:
        assert 0 <= idx < X.shape[0]
    # cluster_centers_ shape
    assert km.cluster_centers_.shape == (n_clusters, X.shape[1])


def test_kmedoids_passes_sklearn_checks() -> None:
    clusterer = KMedoids()
    for estimator, check in estimator_checks_generator(clusterer):
        check(estimator)
