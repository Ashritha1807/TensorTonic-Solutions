import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    points = np.array(points, dtype=float)
    assignments = np.array(assignments)

    d = points.shape[1]
    centroids = []

    for cluster in range(k):
        cluster_points = points[assignments == cluster]

        if len(cluster_points) == 0:
            centroids.append([0.0] * d)
        else:
            centroids.append(cluster_points.mean(axis=0).tolist())

    return centroids