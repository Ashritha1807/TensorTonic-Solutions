import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.asarray(y_train)

    # Find unique classes, their counts, and first occurrence indices
    classes, first_idx, counts = np.unique(
        y_train, return_index=True, return_counts=True
    )

    # Choose class with highest count
    max_count = np.max(counts)
    candidates = np.where(counts == max_count)[0]

    # Tie-break: choose the class that appears first in y_train
    majority_class = classes[candidates[np.argmin(first_idx[candidates])]]

    # Predict majority class for all test samples
    return np.full(len(X_test), majority_class, dtype=int)