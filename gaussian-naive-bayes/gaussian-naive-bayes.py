import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)

    classes = np.unique(y_train)
    eps = 1e-9

    priors = {}
    means = {}
    variances = {}

    # Training
    for c in classes:
        X_c = X_train[y_train == c]

        priors[c] = len(X_c) / len(X_train)
        means[c] = np.mean(X_c, axis=0)

        # Population variance (ddof=0)
        variances[c] = np.var(X_c, axis=0) + eps

    predictions = []

    # Prediction
    for x in X_test:
        log_posteriors = []

        for c in classes:
            log_prior = np.log(priors[c])

            log_likelihood = np.sum(
                -0.5 * np.log(2 * np.pi * variances[c])
                - ((x - means[c]) ** 2) / (2 * variances[c])
            )

            log_posteriors.append(log_prior + log_likelihood)

        predictions.append(int(classes[np.argmax(log_posteriors)]))

    return predictions