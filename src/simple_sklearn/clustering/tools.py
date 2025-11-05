import numpy as np
from scipy.spatial import distance


def calc_zip_distances(set1, set2, distance_metric='euclidean'):
    return [distance.cdist([p1], [p2], metric=distance_metric)[0][0] for p1, p2 in zip(set1, set2)]


def calc_max_zip_distance(set1, set2, distance_metric='euclidean'):
    return np.max(calc_zip_distances(set1, set2, distance_metric))


def calc_min_zip_distance(set1, set2, distance_metric='euclidean'):
    return np.min(calc_zip_distances(set1, set2, distance_metric))


def calc_distance_matrix(points, targets, distance_metric='euclidean'):
    return distance.cdist(points, targets, metric=distance_metric)


def find_closest_point(points, target, distance_metric='euclidean'):
    distances = distance.cdist(points, [target], metric=distance_metric).flatten()
    closest_index = int(np.argmin(distances))
    return closest_index, points[closest_index]
