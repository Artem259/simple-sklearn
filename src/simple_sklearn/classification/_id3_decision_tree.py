"""ID3 Decision Tree Classification.

This module provides the `ID3DecisionTreeClassifier` class and its associated node structures.
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data
from typing_extensions import Self


class ID3DecisionTreeClassifier(ClassifierMixin, BaseEstimator):  # type: ignore
    """Perform Decision Tree classification using the ID3 algorithm.

    The ID3 Decision Tree classifier builds a tree from categorical features by recursively
    partitioning the data based on the feature that maximizes information gain
    (decreases entropy).

    Note:
        This implementation is strictly for discrete/categorical feature data.
        Passing continuous features will result in severe overfitting.

    Attributes:
        classes_: The unique class labels observed in the training data.
        num_features_: The number of features seen during fitting.
        feature_unique_values_: A list of sets containing the unique values encountered for each feature.
        tree_: The root node of the fitted decision tree (an `ID3DecisionTreeNode` instance).
            Nodes store integer-encoded labels mapped to the `classes_` array.
    """

    classes_: NDArray[Any]
    num_features_: int
    feature_unique_values_: list[set[Any]]
    tree_: "ID3DecisionTreeNode"

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: Any, y: Any) -> Self:
        """Fit the ID3 Decision Tree classification model.

        Args:
            X: Training instances to fit the model. Can be an array-like of shape `(n_samples, n_features)`.
            y: Target values (class labels) for the training instances. Array-like of shape `(n_samples,)`.

        Returns:
            The fitted instance.

        Raises:
            ValueError: If `y` is of a continuous type.
        """
        X, y = validate_data(self, X, y)
        X = np.array(X)

        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
        self.classes_, y = np.unique(y, return_inverse=True)

        self.num_features_ = X.shape[1]
        feat_indices = list(range(self.num_features_))
        df = pd.DataFrame(X, columns=feat_indices)
        df["y"] = y

        self.feature_unique_values_ = [set(df[feat_index]) for feat_index in feat_indices]
        self.tree_ = self._id3_algorithm(df, set(feat_indices))

        return self

    def predict(self, X: Any) -> NDArray[Any]:
        """Predict class labels for the given input data.

        Traverses the fitted decision tree for each sample. If an unseen feature value
        is encountered at a split node, the model falls back to predicting the majority
        class label of that specific node.

        Args:
            X: Instances to predict. Can be an array-like of shape `(n_samples, n_features)`.

        Returns:
            An array of shape `(n_samples,)` containing the predicted class labels for each sample.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.array(X)

        y_pred = []
        for x in X:
            curr_node = self.tree_
            while True:
                match curr_node:
                    case ID3LeafNode():
                        y_pred.append(curr_node.label)
                        break
                    case ID3SplitterNode():
                        feat_value = x[curr_node.feat_index]
                        if feat_value in curr_node.children:
                            curr_node = curr_node.children[feat_value]
                        else:
                            y_pred.append(curr_node.majority_label)
                            break
                    case _:
                        raise TypeError(f"Unknown node type encountered: {type(curr_node)}")

        return np.asarray(self.classes_[y_pred])

    def _id3_algorithm(self, df: pd.DataFrame, feat_indices: set[int]) -> "ID3DecisionTreeNode":
        """Recursively build the decision tree using the ID3 algorithm.

        Args:
            df: A pandas.DataFrame containing the training data and target labels for the current node.
            feat_indices: A set of remaining feature indices available for splitting.

        Returns:
            An `ID3DecisionTreeNode` representing the root of the constructed subtree
            (either an `ID3SplitterNode` or `ID3LeafNode`).
        """
        y_counts = df["y"].value_counts()
        most_frequent_y = y_counts.index[0]

        if len(y_counts) == 1 or not feat_indices:
            return ID3LeafNode(label=most_frequent_y)

        best_feat = max(feat_indices, key=lambda x: self._information_gain(df, x))
        node = ID3SplitterNode(feat_index=best_feat, majority_label=most_frequent_y)

        for best_feat_value, df_group in df.groupby(by=best_feat):
            child_node = self._id3_algorithm(df_group, feat_indices - {best_feat})
            node.children[best_feat_value] = child_node

        all_best_feat_values = self.feature_unique_values_[best_feat]
        df_best_feat_values = set(df[best_feat])
        unseen_best_feat_values = all_best_feat_values - df_best_feat_values
        for best_feat_value in unseen_best_feat_values:
            node.children[best_feat_value] = ID3LeafNode(label=most_frequent_y)

        return node

    def _information_gain(self, df: pd.DataFrame, feat_index: int) -> float:
        """Calculate the information gain for a potential feature split.

        Args:
            df: A pandas.DataFrame containing the current node's data and labels.
            feat_index: The index of the feature to evaluate.

        Returns:
            The calculated information gain (reduction in entropy).
        """
        total_entropy = self._calc_entropy(df)
        values = df[feat_index].unique()
        weighted_entropy = 0.0

        for value in values:
            subset = df[df[feat_index] == value]
            weight = len(subset) / len(df)
            weighted_entropy += weight * self._calc_entropy(subset)

        return total_entropy - weighted_entropy

    def _calc_entropy(self, df: pd.DataFrame) -> float:
        """Calculate the Shannon entropy of the target labels in the given data.

        Args:
            df: A pandas.DataFrame containing the target labels in a 'y' column.

        Returns:
            The calculated entropy value.
        """
        value_counts = df["y"].value_counts(normalize=True)
        entropy = float(-np.sum(value_counts * np.log2(value_counts)))
        return entropy


class ID3DecisionTreeNode:
    """Base class for nodes in the decision tree."""

    pass


class ID3SplitterNode(ID3DecisionTreeNode):
    """A decision tree node representing a feature split.

    Args:
        feat_index: The index of the feature used for splitting at this node.
        majority_label: The integer-encoded majority class label at this node,
            used as a fallback during prediction.

    Attributes:
        children: A dictionary mapping feature values to child `ID3DecisionTreeNode` instances.
    """

    def __init__(self, feat_index: int, majority_label: int) -> None:
        self.feat_index: int = feat_index
        self.majority_label: int = majority_label
        self.children: dict[Any, ID3DecisionTreeNode] = {}


class ID3LeafNode(ID3DecisionTreeNode):
    """A decision tree node representing a terminal leaf.

    Args:
        label: The integer-encoded predicted class label for this leaf.
    """

    def __init__(self, label: int) -> None:
        self.label: int = label
