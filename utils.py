from jax import numpy as jnp
import jax
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from project_directories import PROC_DATA_DIR


def load_split_data(test_size=0.2):
    """
    Load and split open hole composite specimen labeled data .

    Parameters
    ----------
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.2.

    Returns
    -------
    tuple of ndarrays
        A tuple containing four NumPy arrays:
        - X_train: Training data with shape (n_train_samples, n_features).
        - X_test: Testing data with shape (n_test_samples, n_features).
        - y_train: Training labels with shape (n_train_samples,).
        - y_test: Testing labels with shape (n_test_samples,).

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = load_split_data(test_size=0.25)
    >>> X_train.shape, X_test.shape, y_train.shape, y_test.shape
    ((800, 3), (200, 3), (800,), (200,))

    Notes
    -----
    This function reads labeled data from a CSV file, extracts the features and labels,
    and then splits the data into training and testing sets using `train_test_split` from
    scikit-learn. The default test size is 0.2, but it can be adjusted using the `test_size` parameter.

    """
    df = pd.read_csv(PROC_DATA_DIR / 'data_labeled.csv')
    X = df[['eps11', 'eps22', 'eps12']].to_numpy()
    y = df['failed'].to_numpy(dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


def square_kernel_matrix(X, kernel_func):
    """
    Compute the square kernel matrix using JAX.

    Parameters
    ----------
    X : jax.numpy.ndarray
        Input data array of shape (n_samples, n_features).
    kernel_func : callable
        Kernel function that takes two arguments and computes the kernel value.

    Returns
    -------
    K : jax.numpy.ndarray
        Square kernel matrix of shape (n_samples, n_samples).
    """
    n_samples = X.shape[0]
    K = jnp.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):
            # Compute kernel value and fill in both (i, j) and (j, i) entries
            K_i_j = kernel_func(X[i], X[j])
            K = K.at[i, j].set(K_i_j)
            K = K.at[j, i].set(K_i_j)

    return K


def target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    """
    Compute the kernel-target alignment using JAX.

    Parameters
    ----------
    X : array_like
        Input data matrix.
    Y : array_like
        Class labels.
    kernel : callable
        Kernel function to compute the square kernel matrix.
    assume_normalized_kernel : bool, optional
        Flag indicating whether the kernel is assumed to be normalized (default is False).
    rescale_class_labels : bool, optional
        Flag indicating whether to rescale class labels (default is True).

    Returns
    -------
    float
        Target alignment value.

    Notes
    -----
    The target alignment is computed as the inner product between the kernel matrix and the outer product of class labels.

    Examples
    --------
    >>> X = ...  # Input data matrix
    >>> Y = ...  # Class labels
    >>> kernel = ...  # Kernel function
    >>> result = target_alignment_qml_jax(X, Y, kernel)
    >>> print(result)
    Target Alignment: 0.1234
    """

    X = jnp.array(X)
    Y = jnp.array(Y)

    K = square_kernel_matrix(
        X,
        kernel,
    )

    if rescale_class_labels:
        nplus = jnp.count_nonzero(Y == 1)
        nminus = len(Y) - nplus
        _Y = jnp.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = jnp.array(Y)

    T = jnp.outer(_Y, _Y)
    inner_product = jnp.sum(K * T)
    norm = jnp.sqrt(jnp.sum(K * K) * jnp.sum(T * T))

    return inner_product / norm
