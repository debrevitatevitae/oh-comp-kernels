from jax import numpy as jnp


def square_kernel_matrix(X, kernel_func):
    """
    Compute the square kernel matrix using JAX. Since this function will be used for quantum kernels, it is assumed that the kernel is normalized (diagonal elements of the kernel matrix are equal to 1).

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
    N = X.shape[0]
    K = jnp.eye(N)

    for i in range(N-1):
        for j in range(i+1, N):
            # Compute kernel value and fill in both (i, j) and (j, i) entries
            K_i_j = kernel_func(X[i], X[j])
            K = K.at[i, j].set(K_i_j)
            K = K.at[j, i].set(K_i_j)

    return K


def frobenius_ip(A, B):
    """
    Compute the Frobenius inner product between two matrices.

    Parameters
    ----------
    A : ndarray
        The first matrix.
    B : ndarray
        The second matrix.

    Returns
    -------
    float
        The Frobenius inner product between the two matrices.

    Examples
    --------
    >>> A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    >>> frobenius_ip(A, B)
    70.0

    Notes
    -----
    The Frobenius inner product between two matrices A and B is defined as the sum of
    the element-wise products of the corresponding entries:

    Frob(A, B) = sum(A[i, j] * B[i, j] for all i, j)

    This function utilizes the JAX library's numpy implementation (jax.numpy).

    """
    return jnp.sum(A * B)


def target_alignment(
    X,
    Y,
    kernel,
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
    >>> result = target_alignment(X, Y, kernel)
    >>> print(result)
    Target Alignment: 0.1234
    """
    N = X.shape[0]

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

    K_target = jnp.outer(_Y, _Y)
    kta = frobenius_ip(K, K_target) / (N * jnp.sqrt(frobenius_ip(K, K)))

    return kta
