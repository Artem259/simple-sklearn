from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance


def calc_zip_distances(set1: NDArray[Any], set2: NDArray[Any]) -> list[float]:
    return [float(distance.cdist([p1], [p2])[0][0]) for p1, p2 in zip(set1, set2, strict=True)]


def calc_max_zip_distance(set1: NDArray[Any], set2: NDArray[Any]) -> float:
    return float(np.max(calc_zip_distances(set1, set2)))


def calc_min_zip_distance(set1: NDArray[Any], set2: NDArray[Any]) -> float:
    return float(np.min(calc_zip_distances(set1, set2)))


def calc_distance_matrix(points: NDArray[Any], targets: NDArray[Any]) -> NDArray[Any]:
    return np.asarray(distance.cdist(points, targets))


def find_closest_point(points: NDArray[Any], target: NDArray[Any]) -> tuple[int, NDArray[Any]]:
    distances = distance.cdist(points, [target]).flatten()
    closest_index = int(np.argmin(distances))
    return closest_index, points[closest_index]
