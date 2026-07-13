import numpy as np
import pytest
from numpy.typing import NDArray

from simple_sklearn.classification import DecisionTreeClassifier
from simple_sklearn.classification.tree_structure import (
    DecisionTreeNode,
    LeafNode,
    SplitterNode,
)
from tests.classification.testing_utils import assert_valid_classification_predictions
from tests.testing_utils import assert_attributes_match_types


def test_decision_tree_fit_predict(dataset_4f: tuple[NDArray[np.int64], NDArray[np.str_]]) -> None:
    X, y = dataset_4f
    n_features = X.shape[1]
    unique_classes = set(y)
    n_classes = len(unique_classes)

    assert 3 not in X[:, 0], "Value 3 is present in the first feature of X"
    assert 2 not in X[:, 1], "Value 2 is present in the second feature of X"
    X_pred = np.array(
        [
            [2, 1, 1, 1],
            [0, 1, 0, 0],
            [3, 1, 1, 1],  # unknown value in the first feature
            [0, 2, 1, 1],  # unknown value in the second feature
            [3, 2, 1, 1],  # unknown value in the first and second features
        ]
    )

    expected_attributes = {
        "classes_": NDArray[np.str_],
        "num_features_": int,
        "feature_unique_values_": list[set[int]],
        "tree_": DecisionTreeNode,
    }

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    # Validate expected learned attributes
    assert_attributes_match_types(clf, expected_attributes)

    # Validate classes_ attribute
    assert clf.classes_.shape[0] == n_classes
    assert set(clf.classes_) == unique_classes

    # Validate num_features_ attribute
    assert clf.num_features_ == n_features

    # Validate feature_unique_values_ attribute
    assert len(clf.feature_unique_values_) == n_features
    for i, unique_vals in enumerate(clf.feature_unique_values_):
        assert unique_vals == set(X[:, i])

    # Validate tree_ attribute
    # The root must resolve to either a split or a terminal leaf
    assert isinstance(clf.tree_, (SplitterNode, LeafNode))

    # Check predictions
    y_pred = clf.predict(X_pred)
    assert_valid_classification_predictions(X_pred, y_pred, y, clf.classes_)
    assert list(y_pred) == ["y1", "y1", "y1", "y0", "y1"]


def test_decision_tree_predict_unknown_node() -> None:
    clf = DecisionTreeClassifier()
    clf.fit([[0, 0]], [0])

    class DummyNode(DecisionTreeNode):
        pass

    clf.tree_ = DummyNode()

    with pytest.raises(TypeError, match="Unknown node type encountered"):
        clf.predict([[0, 0]])
