import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from project_directories import PROC_DATA_DIR


def rbf_kernel(x, y, gamma=1.):
    """
    Compute the Gaussian kernel between two vectors.

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
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
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

    df = pd.read_csv(PROC_DATA_DIR / 'data_labeled.csv')
    X = df[['eps11', 'eps22', 'eps12']].to_numpy()
    y = df['failed'].to_numpy(dtype=np.int32)

    X = np.array(X, requires_grad=False)
    y = np.array(y, requires_grad=False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kta = qml.kernels.target_alignment(
        X_scaled, y, rbf_kernel, assume_normalized_kernel=True)

    print(kta)
