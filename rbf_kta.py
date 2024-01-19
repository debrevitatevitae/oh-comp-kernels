import math
import os
import time
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn.preprocessing import StandardScaler

from ohqk.project_directories import RESULTS_DIR
from ohqk.utils import load_split_data


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
    start = time.time()

    np.random.seed(42)

    # data loading, splitting and scaling
    X_train, X_test, y_train, y_test = load_split_data(test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # declare some batch sizes and number of repetitions
    N = len(y_train)
    batch_sizes = [math.ceil(frac*N)
                   for frac in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]]
    num_reps = 100

    # create a pd.DataFrame with the batch sizes as columns. Rows will be the repetitions
    columns = [f'Batch Size {size}' for size in batch_sizes]
    rows = [f'Rep no {n}' for n in range(num_reps)]
    df = pd.DataFrame(columns=columns, index=rows)

    for batch_size in batch_sizes:
        for n in range(num_reps):
            # select some indices at random for the batch and isolate the batch
            idxs_batch = np.random.choice(N, size=batch_size)
            X_train_scaled_batch = X_train_scaled[idxs_batch]
            y_train_batch = y_train[idxs_batch]

            # compute the KTA for this batch
            kta = qml.kernels.target_alignment(
                X_train_scaled_batch, y_train_batch, rbf_kernel, assume_normalized_kernel=True)

            # store the kta value in the DataFrame
            df.loc[f'Rep no {n}', f'Batch Size {batch_size}'] = kta

            # print KTA for this iteration
            print(f'Batch size {batch_size:d}, rep no {n:d}, kta {kta:.5f}')

    kta_full_train_set = qml.kernels.target_alignment(
        X_train_scaled, y_train, rbf_kernel, assume_normalized_kernel=True)
    print(f"Full training set, kta {kta_full_train_set:.5f}")

    # save DataFrame to a csv file named after this Python file
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    df.to_csv(RESULTS_DIR / f'{python_file_name_no_ext}.csv', index=False)

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
