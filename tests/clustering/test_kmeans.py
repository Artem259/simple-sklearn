import numpy as np
import numpy.testing as npt
import sklearn
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.utils.estimator_checks import estimator_checks_generator

from simple_sklearn.clustering import KMeans


def test_kmeans_matches_sklearn_on_small_dataset() -> None:
    # three well-separated clusters
    X = np.vstack(
        [
            np.random.RandomState(0).normal(loc=[0, 0], scale=0.1, size=(5, 2)),
            np.random.RandomState(1).normal(loc=[5, 5], scale=0.1, size=(5, 2)),
            np.random.RandomState(2).normal(loc=[10, 0], scale=0.1, size=(5, 2)),
        ]
    )
    init = np.array([[0, 0], [5, 5], [10, 0]])
    n_clusters = 3
    max_iter = 10
    random_state = 42

    custom = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=random_state)
    custom.fit(X)

    sk = sklearn.cluster.KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=random_state)
    sk.fit(X)

    # labels may be permuted but adjusted_rand_score should be 1.0
    ars = adjusted_rand_score(custom.labels_, sk.labels_)
    assert ars == 1.0

    # inertia should be close
    npt.assert_allclose(custom.inertia_, sk.inertia_, rtol=1e-6, atol=1e-8)
    # check n_iter_ exists
    assert hasattr(custom, "n_iter_")


def test_kmeans_passes_sklearn_checks() -> None:
    clusterer = KMeans()
    for estimator, check in estimator_checks_generator(clusterer):
        check(estimator)
