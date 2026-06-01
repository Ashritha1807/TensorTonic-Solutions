import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute unnormalized log posteriors for Bernoulli Naive Bayes.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    classes = np.unique(y_train)
    n_test = X_test.shape[0]
    n_features = X_train.shape[1]

    log_posteriors = np.zeros((n_test, len(classes)))

    for idx, cls in enumerate(classes):
        X_cls = X_train[y_train == cls]
        n_cls = len(X_cls)

        # Log prior
        log_prior = np.log(n_cls / len(y_train))

        # Bernoulli likelihood with Laplace smoothing
        theta = (X_cls.sum(axis=0) + 1) / (n_cls + 2)

        # Log posterior
        log_posteriors[:, idx] = (
            log_prior +
            (X_test * np.log(theta)).sum(axis=1) +
            ((1 - X_test) * np.log(1 - theta)).sum(axis=1)
        )

    return log_posteriors