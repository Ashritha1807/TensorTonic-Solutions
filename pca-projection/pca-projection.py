import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.array(X, dtype=float)

    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Sample covariance matrix
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Top-k principal components
    W = eigenvectors[:, :k]

    # Project data
    X_proj = X_centered @ W

    return X_proj.tolist()