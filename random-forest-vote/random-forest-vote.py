import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    predictions = np.array(predictions)

    n_samples = predictions.shape[1]
    result = []

    for i in range(n_samples):
        votes = predictions[:, i]
        counts = np.bincount(votes)
        result.append(int(np.argmax(counts)))  # smallest label wins ties

    return result