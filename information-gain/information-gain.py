import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)

    y_left = y[split_mask]
    y_right = y[~split_mask]

    n_left = len(y_left)
    n_right = len(y_right)
    n_total = len(y)

    # Edge case: empty partition
    if n_left == 0 or n_right == 0:
        return 0.0

    parent_entropy = _entropy(y)
    left_entropy = _entropy(y_left)
    right_entropy = _entropy(y_right)

    weighted_entropy = (
        (n_left / n_total) * left_entropy +
        (n_right / n_total) * right_entropy
    )

    return float(parent_entropy - weighted_entropy)