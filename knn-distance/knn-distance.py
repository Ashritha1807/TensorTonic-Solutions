import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    # Handle 1D inputs
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Pairwise squared Euclidean distances
    diff = X_test[:, None, :] - X_train[None, :, :]
    distances = np.sum(diff ** 2, axis=2)

    neighbors = np.argsort(distances, axis=1)

    if k <= n_train:
        return neighbors[:, :k].astype(int)

    # Pad with -1 if k > n_train
    result = np.full((n_test, k), -1, dtype=int)
    result[:, :n_train] = neighbors
    return result