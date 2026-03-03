import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class DecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
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

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.array(X)

        y_pred = []
        for x in X:
            curr_node = self.tree_
            while not curr_node.is_leaf:
                feat_value = x[curr_node.feat_index]
                if feat_value in curr_node.children:
                    curr_node = curr_node.children[feat_value]
                else:
                    y_pred.append(curr_node.majority_label)
                    break
            else:
                y_pred.append(curr_node.label)

        return self.classes_[y_pred]

    def _id3_algorithm(self, df: pd.DataFrame, feat_indices: set[int]) -> "DecisionTreeNode":
        node = DecisionTreeNode()

        y_counts = df["y"].value_counts()
        most_frequent_y = y_counts.index[0]
        node.majority_label = most_frequent_y

        if len(y_counts) == 1 or not feat_indices:
            node.make_leaf(most_frequent_y)
            return node

        best_feat = max(feat_indices, key=lambda x: self._information_gain(df, x))
        node.make_splitter(best_feat)

        for best_feat_value, df_group in df.groupby(by=best_feat):
            child_node = self._id3_algorithm(df_group, feat_indices - {best_feat})
            node.add_splitter_child(best_feat_value, child_node)

        all_best_feat_values = self.feature_unique_values_[best_feat]
        df_best_feat_values = set(df[best_feat])
        unseen_best_feat_values = all_best_feat_values - df_best_feat_values
        for best_feat_value in unseen_best_feat_values:
            child_node = DecisionTreeNode()
            child_node.make_leaf(most_frequent_y)
            node.add_splitter_child(best_feat_value, child_node)

        return node

    def _information_gain(self, df: pd.DataFrame, feat_index: int) -> float:
        total_entropy = self._calc_entropy(df)
        values = df.iloc[:, feat_index].unique()
        weighted_entropy = 0.0

        for value in values:
            subset = df[df.iloc[:, feat_index] == value]
            weight = len(subset) / len(df)
            weighted_entropy += weight * self._calc_entropy(subset)

        return total_entropy - weighted_entropy

    def _calc_entropy(self, df: pd.DataFrame) -> float:
        value_counts = df["y"].value_counts(normalize=True)
        entropy = -np.sum(value_counts * np.log2(value_counts))
        return entropy


class DecisionTreeNode:
    def __init__(self):
        self.is_leaf = None

        # For splitter node
        self.feat_index = None
        self.children = None

        # For leaf node
        self.label = None

        # For unknown-value fallback
        self.majority_label = None

    def make_splitter(self, feat_index: int):
        self.is_leaf = False
        self.feat_index = feat_index
        self.children = {}
        self.label = None

    def make_leaf(self, label: int):
        self.is_leaf = True
        self.feat_index = None
        self.children = None
        self.label = label
        self.majority_label = label

    def add_splitter_child(self, feat_value, child_node: "DecisionTreeNode"):
        self.children[feat_value] = child_node
