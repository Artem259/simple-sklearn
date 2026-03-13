import sklearn

from simple_sklearn.classification.one_r import OneRClassifier
from simple_sklearn.classification.naive_bayes import NaiveBayesClassifier
from simple_sklearn.classification.decision_tree import DecisionTreeClassifier, DecisionTreeNode, SplitterNode, LeafNode
from simple_sklearn.classification.k_neighbors import KNeighborsClassifier


def OneRClassifier_info(clf: OneRClassifier):
    for rule in clf.prediction_rules_.items():
        print(f"x[{clf.best_feature_index_}] == {rule[0]}: y = {clf.classes_[rule[1]]}")


def NaiveBayesClassifier_info(clf: NaiveBayesClassifier):
    info = {
        "class_log_prior_": clf.class_log_prior_,
        "feature_log_prob_": clf.feature_log_prob_,
        "feature_unknown_log_probs_": clf.feature_unknown_log_probs_,
    }
    for key, value in info.items():
        print(f"{key}:")
        print(*value, "\n")


def sklearn_CategoricalNB_info(clf: sklearn.naive_bayes.CategoricalNB):
    info = {
        "class_log_prior_": clf.class_log_prior_,
        "feature_log_prob_": clf.feature_log_prob_,
    }
    for key, value in info.items():
        print(f"{key}:")
        print(*value, "\n")


def DecisionTreeClassifier_info(clf: DecisionTreeClassifier):
    def display_tree(node: DecisionTreeNode, prefix="", is_last=True, is_root=False):
        connector = ""
        new_prefix = ""

        if not is_root:
            connector = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")

        if isinstance(node, LeafNode):
            print(prefix + connector + f"Label: {node.label}")
        if isinstance(node, SplitterNode):
            print(prefix + connector + f"Feature[{node.feat_index + 1}]")

            children_items = list(node.children.items())
            for i, (feat_value, child) in enumerate(children_items):
                is_last_child = (i == len(children_items) - 1)
                print(new_prefix + f"({feat_value})")
                display_tree(child, prefix=new_prefix, is_last=is_last_child)

    display_tree(clf.tree_)


def KNeighborsClassifier_info(clf: KNeighborsClassifier, X_pred):
    neigh_dist, neigh_ind = clf.kneighbors(X_pred)
    info = {
        "neighbors_dist": neigh_dist,
        "neighbors_indices": neigh_ind,
    }
    for key, value in info.items():
        print(f"{key}:")
        print(*value, "\n")


def sklearn_KNeighborsClassifier_info(clf: sklearn.neighbors.KNeighborsClassifier, X_pred):
    neigh_dist, neigh_ind = clf.kneighbors(X_pred)
    info = {
        "neighbors_dist": neigh_dist,
        "neighbors_indices": neigh_ind,
    }
    for key, value in info.items():
        print(f"{key}:")
        print(*value, "\n")
