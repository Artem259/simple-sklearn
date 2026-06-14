"""Distance Calculation Tools.

This module provides utility functions for calculating spatial distances.
"""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance


def calc_zip_distances(set1: NDArray[Any], set2: NDArray[Any]) -> NDArray[Any]:
    """Calculate the element-wise pairwise distances between two sets of points.

    Computes the Euclidean distance between each aligned pair of points
    `(set1[i], set2[i])`. Both input arrays must have the same length.

    Args:
        set1: The first array of points of shape `(n_samples, n_features)`.
        set2: The second array of points of shape `(n_samples, n_features)`.

    Returns:
        An array of distances corresponding to each pair of points.
    """
    distances = np.linalg.norm(set1 - set2, axis=1)
    return cast(NDArray[Any], distances)


def calc_max_zip_distance(set1: NDArray[Any], set2: NDArray[Any]) -> float:
    """Calculate the maximum element-wise distance between two sets of points.

    Evaluates the element-wise pairwise distances between aligned points in
    `set1` and `set2` and returns the maximum observed distance.

    Args:
        set1: The first array of points of shape `(n_samples, n_features)`.
        set2: The second array of points of shape `(n_samples, n_features)`.

    Returns:
        The maximum distance found among the aligned point pairs.
    """
    return float(np.max(calc_zip_distances(set1, set2)))


def calc_min_zip_distance(set1: NDArray[Any], set2: NDArray[Any]) -> float:
    """Calculate the minimum element-wise distance between two sets of points.

    Evaluates the element-wise pairwise distances between aligned points in
    `set1` and `set2` and returns the minimum observed distance.

    Args:
        set1: The first array of points of shape `(n_samples, n_features)`.
        set2: The second array of points of shape `(n_samples, n_features)`.

    Returns:
        The minimum distance found among the aligned point pairs.
    """
    return float(np.min(calc_zip_distances(set1, set2)))


def calc_distance_matrix(points: NDArray[Any], targets: NDArray[Any]) -> NDArray[Any]:
    """Calculate the full pairwise distance matrix between two sets of points.

    Computes the Euclidean distance between every point in the `points` array
    and every point in the `targets` array.

    Args:
        points: An array of points of shape `(n_samples_1, n_features)`.
        targets: An array of target points of shape `(n_samples_2, n_features)`.

    Returns:
        A 2D array of shape `(n_samples_1, n_samples_2)` where the element at
        index `[i, j]` is the distance between `points[i]` and `targets[j]`.
    """
    return np.asarray(distance.cdist(points, targets))


def find_closest_point(points: NDArray[Any], target: NDArray[Any]) -> tuple[int, NDArray[Any]]:
    """Find the closest point in a dataset to a given target point.

    Computes the distances from all candidate `points` to a single `target`
    point and identifies the nearest neighbor.

    Args:
        points: An array of candidate points of shape `(n_samples, n_features)`.
        target: The reference target point of shape `(n_features,)`.

    Returns:
        index: The integer index of the closest point in the `points` array.
        closest_point: The coordinates of the closest point itself.
    """
    distances = np.linalg.norm(points - target, axis=1)
    closest_index = int(np.argmin(distances))
    return closest_index, points[closest_index]
