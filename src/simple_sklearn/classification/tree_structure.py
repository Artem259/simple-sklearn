"""Node structures for representing decision trees.

This module exposes the node classes and type aliases used to construct and traverse
decision trees, including the base splitter interface, algorithm-specific splitters,
and terminal prediction leaves.
"""

from ._base_tree_structure import DecisionTreeNode, LeafNode, SplitterNode
from ._id3_decision_tree import ID3DecisionTreeNode, ID3SplitterNode

__all__ = [
    "DecisionTreeNode",
    "ID3DecisionTreeNode",
    "ID3SplitterNode",
    "LeafNode",
    "SplitterNode",
]
