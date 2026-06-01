import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    points = np.array(points)
    centroids = np.array(centroids)

    assignments = []

    for point in points:
        distances = np.sum((centroids - point) ** 2, axis=1)
        assignments.append(int(np.argmin(distances)))

    return assignments