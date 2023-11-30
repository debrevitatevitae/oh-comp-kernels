import pennylane as qml
from pennylane import numpy as np
from sklearn.preprocessing import StandardScaler

from utils import load_split_data, target_alignment


def rbf_kernel(x, y, gamma=1.):
    """
    Compute the Gaussian kernel between two vectors. Uses jax.

    Parameters
    ----------
    x, y : array_like
        Input vectors.
    gamma : float, optional
        Kernel parameter (default is 1.0).

    Returns
    -------
    float
        Gaussian kernel value.

    Notes
    -----
    The Gaussian kernel is defined as:

    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    where sigma is calculated as 1 / sqrt(2 * gamma).

    Examples
    --------
    >>> x = jnp.array([1, 2, 3])
    >>> y = jnp.array([4, 5, 6])
    >>> gamma_value = 0.1
    >>> result = gaussian_kernel(x, y, gamma=gamma_value)
    >>> print(result)
    Gaussian Kernel: 0.9875774860176064
    """
    diff = np.subtract(x, y)
    norm_squared = np.dot(diff, diff)
    return np.exp(-gamma*norm_squared)


if __name__ == '__main__':
    np.random.seed(42)

    X_train, X_test, y_train, y_test = load_split_data(test_size=0.9)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    kta_initial = qml.kernels.target_alignment(
        X_train_scaled, y_train, rbf_kernel, assume_normalized_kernel=True)

    print(kta_initial)
