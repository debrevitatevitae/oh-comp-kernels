import jax
from jax import numpy as jnp
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from project_directories import PROC_DATA_DIR
from utils import target_alignment


@jax.jit
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
    diff = jnp.subtract(x, y)
    norm_squared = jnp.dot(diff, diff)
    return jnp.exp(-gamma*norm_squared)


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)

    np.random.seed(42)

    df = pd.read_csv(PROC_DATA_DIR / 'data_labeled.csv')
    X = df[['eps11', 'eps22', 'eps12']].to_numpy()
    y = df['failed'].to_numpy(dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    kta_initial = target_alignment(
        X_train_scaled, y_train, rbf_kernel, assume_normalized_kernel=True)

    print(kta_initial)
