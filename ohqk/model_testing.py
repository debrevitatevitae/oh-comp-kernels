import numpy as np


def produce_clf_learning_curve(
    clf,
    X_train,
    X_test,
    y_train,
    y_test,
    train_fractions=np.linspace(0.1, 1, 10),
):
    """Produce a learning curve for a given classifier.

    Parameters
    ----------
    clf : sklearn.base.BaseEstimator
        A classifier with a `fit` and `predict` method.
    X_train : np.ndarray
        Training data.
    X_test : np.ndarray
        Test data.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    train_fractions : np.ndarray
        Fractions of the training data to use for the learning curve.

    Returns
    -------
    train_scores : np.ndarray
        Training scores for each fraction.
    test_scores : np.ndarray
        Test scores for each fraction.
    """
    train_scores = []
    test_scores = []

    for fraction in train_fractions:
        n_train = int(fraction * X_train.shape[0])
        X_train_subset = X_train[:n_train]
        y_train_subset = y_train[:n_train]

        clf.fit(X_train_subset, y_train_subset)

        train_score = clf.score(X_train_subset, y_train_subset)
        test_score = clf.score(X_test, y_test)

        train_scores.append(train_score)
        test_scores.append(test_score)

    return np.array(train_scores), np.array(test_scores)
