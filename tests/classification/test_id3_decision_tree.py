import numpy as np
from numpy.typing import NDArray

from simple_sklearn.classification import ID3DecisionTreeClassifier
from simple_sklearn.classification.tree_structure import (
    ID3DecisionTreeNode,
    LeafNode,
)
from tests.classification.testing_utils import assert_valid_classification_predictions
from tests.testing_utils import assert_attributes_match_types, assert_matches_type


def test_id3_decision_tree_fit_predict(dataset_4f: tuple[NDArray[np.int64], NDArray[np.str_]]) -> None:
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
        "tree_": ID3DecisionTreeNode,
    }

    clf = ID3DecisionTreeClassifier()
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
    # Traverse tree recursively and check each node
    def assert_valid_id3_tree(
        node: ID3DecisionTreeNode,
        used_features: frozenset[int] = frozenset(),
    ) -> None:
        assert_matches_type(node, ID3DecisionTreeNode)

        if isinstance(node, LeafNode):
            assert 0 <= node.label < len(unique_classes)
            return

        assert 0 <= node.majority_label < len(unique_classes)
        assert 0 <= node.feat_index < n_features
        assert node.feat_index not in used_features
        assert set(node.children.keys()) == clf.feature_unique_values_[node.feat_index]

        branches = list(node.iter_branches())
        assert dict(branches) == node.children
        for _, child_node in branches:
            assert_valid_id3_tree(child_node, used_features | {node.feat_index})

    assert_valid_id3_tree(clf.tree_)

    # Check predictions
    y_pred = clf.predict(X_pred)
    assert_valid_classification_predictions(X_pred, y_pred, y, clf.classes_)
    assert list(y_pred) == ["y1", "y1", "y1", "y0", "y1"]
