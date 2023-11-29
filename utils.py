from jax import numpy as jnp
import jax


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
