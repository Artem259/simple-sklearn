"""Supervised machine learning algorithms for classification tasks.

This subpackage contains implementations of standard classification models.
All estimators in this module inherit from
[`sklearn.base.BaseEstimator`][] and [`sklearn.base.ClassifierMixin`][].

Available models:
 - [`OneRClassifier`][simple_sklearn.classification.OneRClassifier]:
    1R (One Rule) classification.
 - [`NaiveBayesClassifier`][simple_sklearn.classification.NaiveBayesClassifier]:
    Categorical Naive Bayes classification.
 - [`KNeighborsClassifier`][simple_sklearn.classification.KNeighborsClassifier]:
    K-Nearest Neighbors classification.
 - [`DecisionTreeClassifier`][simple_sklearn.classification.DecisionTreeClassifier]:
    Decision Tree classification using the ID3 algorithm.
"""

from ._decision_tree import DecisionTreeClassifier
from ._k_neighbors import KNeighborsClassifier
from ._naive_bayes import NaiveBayesClassifier
from ._one_r import OneRClassifier

__all__ = [
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
    "NaiveBayesClassifier",
    "OneRClassifier",
]
