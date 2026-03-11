import numpy as np
from sklearn.utils.estimator_checks import estimator_checks_generator

from simple_sklearn.classification.decision_tree import DecisionTreeClassifier


def test_decision_tree_predict_and_tree_structure() -> None:
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 1],
        ]
    )
    y = np.array([0, 0, 1, 1, 1])

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    # predictions shape and valid classes
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape
    for v in y_pred:
        assert v in np.unique(y)

    # the classifier should expose a tree_ attribute (not None)
    assert hasattr(clf, "tree_")
    assert clf.tree_ is not None


def test_decision_tree_handles_unknown_values() -> None:
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    # Test data includes unseen feature values (e.g. 2)
    X_pred = np.array(
        [
            [2, 0],  # unseen in first feature
            [1, 2],  # unseen in second feature
            [2, 2],  # unseen in both features
        ]
    )
    y_pred = clf.predict(X_pred)

    # Predictions should have same length as test samples
    assert y_pred.shape[0] == X_pred.shape[0]

    # All predictions should be valid known classes
    for p in y_pred:
        assert p in np.unique(y)


def test_decision_tree_passes_sklearn_checks() -> None:
    classifier = DecisionTreeClassifier()
    for estimator, check in estimator_checks_generator(classifier):
        check(estimator)
