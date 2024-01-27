import os
from jax import numpy as jnp
import jax
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ohqk.project_directories import PROC_DATA_DIR, RESULTS_DIR


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


def find_and_sort_files(embedding, trained=None):
    """Finds and sorts files based on the given embedding and trained status."""
    if trained is None:
        files = [f for f in os.listdir(
            RESULTS_DIR) if embedding in f and f.endswith(".csv")]
    else:
        files = [f for f in os.listdir(
            RESULTS_DIR) if embedding in f and f"trained_{trained}" in f and f.endswith(".csv")]

    # for all the previous list, search for the string "wxdy", where x and y are integers and sort the list first by x and then by y
    sorted_files = sorted(
        files,
        key=lambda f: (
            int(f[f.find("w") + 1:f.find("w") + 2]),
            int(f[f.find("d") + 1:f.find("d") + 2]),
        ),
    )

    return sorted_files


def find_order_concatenate_cv_result_files():
    """For each embedding, finds the related cross-validation results files, then orders them by qubit count and number of layers and finally concatenates them into a single list and returns the list."""
    results_iqp_files = find_and_sort_files("iqp")
    results_he2_untrained_files = find_and_sort_files("he2", False)
    results_he2_trained_files = find_and_sort_files("he2", True)

    results_files = results_iqp_files + \
        results_he2_untrained_files + results_he2_trained_files

    return results_files
