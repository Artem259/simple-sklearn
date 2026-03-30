import numpy as np
from sklearn.utils.estimator_checks import estimator_checks_generator

from simple_sklearn.classification import OneRClassifier


def test_one_r_basic_rule_and_predict() -> None:
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
        ]
    )
    # y depends on X[:,0] only
    y = np.array(["class1", "class1", "class2", "class2", "class1"])

    clf = OneRClassifier()
    clf.fit(X, y)

    # Basic attribute checks
    assert hasattr(clf, "best_feature_index_")
    assert hasattr(clf, "prediction_rules_")
    assert isinstance(clf.best_feature_index_, int)

    # Expect best feature to be 0 (first column)
    assert clf.best_feature_index_ == 0

    # Predict on a small set
    X_pred = np.array([[0, 2], [1, 3]])
    y_pred = clf.predict(X_pred)
    assert list(y_pred) == ["class1", "class2"]

    # predictions are members of training classes
    for p in y_pred:
        assert p in clf.classes_


def test_one_r_handles_unknown_values() -> None:
    X = np.array(
        [
            [0, 0],
            [1, 2],
            [2, 1],
            [0, 2],
        ]
    )
    y = np.array(["A", "B", "C", "A"])

    clf = OneRClassifier()
    clf.fit(X, y)

    # Ensure fallback class is set (majority class from training)
    assert hasattr(clf, "fallback_class_")
    assert clf.fallback_class_ in clf.classes_
    assert clf.fallback_class_ == "A"

    # Test includes an unseen value "4" in the best feature column
    X_pred = np.array(
        [
            [4, 0],
            [0, 5],
            [2, 6],
        ]
    )
    y_pred = clf.predict(X_pred)

    # Verify we got predictions for all rows
    assert len(y_pred) == len(X_pred)

    # Check that known values map correctly
    assert y_pred[1] in clf.classes_
    assert y_pred[2] in clf.classes_

    # Check that unseen value "4" got fallback class
    assert y_pred[0] == clf.fallback_class_


def test_one_r_passes_sklearn_checks() -> None:
    classifier = OneRClassifier()
    for estimator, check in estimator_checks_generator(classifier):
        check(estimator)
