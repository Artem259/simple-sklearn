"""Base node structures for decision trees."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass
class SplitterNode(ABC):
    """Abstract base class for decision tree splitting nodes.

    Attributes:
        feat_index: The index of the feature used for splitting at this node.
        majority_label: The integer-encoded majority class label at this node,
            used as a fallback during prediction.
    """

    feat_index: int
    majority_label: int

    @abstractmethod
    def iter_branches(self) -> Iterator[tuple[Any, "DecisionTreeNode"]]:
        """Yield pairs of `(branch_condition, child_node)` for tree traversal.

        Returns:
            An iterator of tuples where the first element describes the branch
                condition and the second element is the child `DecisionTreeNode`.
        """
        raise NotImplementedError


@dataclass
class LeafNode:
    """A decision tree node representing a terminal leaf.

    Attributes:
        label: The integer-encoded predicted class label for this leaf.
    """

    label: int


DecisionTreeNode: TypeAlias = SplitterNode | LeafNode
"""Type alias for any node in a decision tree (`SplitterNode` or `LeafNode`)."""
