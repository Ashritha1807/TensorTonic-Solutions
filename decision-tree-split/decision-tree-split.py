import numpy as np

def decision_tree_split(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    n_samples, n_features = X.shape

    def gini(labels):
        if labels.size == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        return 1.0 - np.sum(p ** 2)

    parent_gini = gini(y)

    best_gain = -1.0
    best_feature = 0
    best_threshold = 0.0

    for feature in range(n_features):
        values = np.unique(X[:, feature])

        if values.size < 2:
            continue

        thresholds = (values[:-1] + values[1:]) / 2.0

        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if not left_mask.any() or not right_mask.any():
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            weighted_gini = (
                (len(y_left) / n_samples) * gini(y_left)
                + (len(y_right) / n_samples) * gini(y_right)
            )

            gain = parent_gini - weighted_gini

            if (
                gain > best_gain or
                (
                    np.isclose(gain, best_gain) and
                    (
                        feature < best_feature or
                        (
                            feature == best_feature and
                            threshold < best_threshold
                        )
                    )
                )
            ):
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return [int(best_feature), float(best_threshold)]