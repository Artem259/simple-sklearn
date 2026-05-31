from typing import Any

from sklearn.utils.estimator_checks import parametrize_with_checks

from simple_sklearn.classification import (
    DecisionTreeClassifier,
    KNeighborsClassifier,
    NaiveBayesClassifier,
    OneRClassifier,
)
from simple_sklearn.clustering import (
    DBSCAN,
    AgglomerativeClustering,
    KMeans,
    KMedoids,
)


@parametrize_with_checks(  # type: ignore[untyped-decorator]
    [
        # classifiers
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        NaiveBayesClassifier(),
        OneRClassifier(),
        # clusterers
        DBSCAN(),
        AgglomerativeClustering(),
        KMeans(),
        KMedoids(),
    ]
)
def test_sklearn_estimator_check(estimator: Any, check: Any) -> None:
    check(estimator)
