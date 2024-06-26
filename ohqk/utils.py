import os
import re
from types import NoneType

import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from ohqk.project_directories import PROC_DATA_DIR, RESULTS_DIR


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1


def relabel_to_m1p1(
    y: np.ndarray | jnp.ndarray | torch.Tensor,
) -> np.ndarray | jnp.ndarray | torch.Tensor:
    return 2 * y - 1


def load_split_scale_data(
    test_size: float = 0.2,
    scale: str | NoneType = None,
    to_torch: bool = False,
    to_jax: bool = False,
):
    """
    Loads, splits, and scales the data. Returns the train and test sets.

    Args:
        test_size (float, optional): The size of the test set. Defaults to 0.2.
        scale (str, optional): The scaling method. If "standard", then the data is scaled using `sklearn.preprocessing.StandardScaler`, with 0 mean and unit variance. If "angle", then the data is scaled using `sklearn.preprocessing.MinMaxScaler`, with values in the range [0, pi]. Defaults to "standard".
        to_torch (bool, optional): If True, then the data is converted to `torch.Tensor` objects. Defaults to False.
        to_jax (bool, optional): If True, then the data is converted to `jax.numpy` arrays. Defaults to False.

    Returns:
        tuple: A tuple containing the train and test sets in the following order: X_train, X_test, y_train, y_test.
    """
    # data loading, splitting, and scaling
    df_data = pd.read_csv(PROC_DATA_DIR / "data_labeled.csv")
    df_train, df_test = train_test_split(df_data, test_size=test_size)

    X_train = df_train[["eps11", "eps22", "eps12"]].to_numpy()
    y_train = df_train["failed"].to_numpy(dtype=np.int32)
    X_test = df_test[["eps11", "eps22", "eps12"]].to_numpy()
    y_test = df_test["failed"].to_numpy(dtype=np.int32)

    match scale:
        case "standard":
            scaler = StandardScaler()
        case "angle":
            scaler = MinMaxScaler(feature_range=(0, np.pi))
        case _:
            scaler = IdentityTransformer()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if to_torch:
        X_train_scaled = torch.tensor(X_train_scaled, requires_grad=False)
        y_train = torch.tensor(y_train, requires_grad=False)
        X_test_scaled = torch.tensor(X_test_scaled, requires_grad=False)
        y_test = torch.tensor(y_test, requires_grad=False)

    if to_jax:
        X_train_scaled = jnp.array(X_train_scaled)
        y_train = jnp.array(y_train)
        X_test_scaled = jnp.array(X_test_scaled)
        y_test = jnp.array(y_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def match_shape_to_num_qubits(X: np.ndarray | jnp.ndarray, num_qubits: int):
    """Matches the shape of the given array to the given number of qubits by
    cycling the columns of the array.

    Args:
        X (np.ndarray|jnp.ndarray): The array to be reshaped.
        num_qubits (int): The number of qubits.

    Returns:
        np.ndarray|jnp.ndarray: The reshaped array.
    """
    X_extended = np.hstack(
        [X[:, i % X.shape[1]].reshape(-1, 1) for i in range(num_qubits)]
    )
    if isinstance(X, jnp.ndarray):
        X_extended = jnp.array(X_extended)

    return X_extended


def find_and_sort_files(embedding, trained=None):
    """Finds and sorts files based on the given embedding and trained
    status."""
    if trained is None:
        files = [
            f
            for f in os.listdir(RESULTS_DIR)
            if embedding in f and f.endswith(".csv")
        ]
    else:
        files = [
            f
            for f in os.listdir(RESULTS_DIR)
            if embedding in f
            and f"trained_{trained}" in f
            and f.endswith(".csv")
        ]

    # for all the previous list, search for the string "wxdy", where x and y are integers and sort the list first by x and then by y
    sorted_files = sorted(
        files,
        key=lambda f: (
            int(f[f.find("w") + 1 : f.find("w") + 2]),
            int(f[f.find("d") + 1 : f.find("d") + 2]),
        ),
    )

    return sorted_files


def find_order_concatenate_cv_result_files():
    """For each embedding, finds the related cross-validation results files,
    then orders them by qubit count and number of layers and finally
    concatenates them into a single list and returns the list."""
    results_iqp_files = find_and_sort_files("iqp")
    results_he2_untrained_files = find_and_sort_files("he2", False)
    results_he2_trained_files = find_and_sort_files("he2", True)

    results_files = (
        results_iqp_files
        + results_he2_untrained_files
        + results_he2_trained_files
    )

    return results_files


def get_info_from_results_file_name(
    results_file: str,
    embedding_names=["iqp", "he2"],
):
    """Given a results file name, returns the embedding type, the number of
    qubits and the number of layers."""
    # Use a regular expression to search in the file name for the embedding
    # name.
    embedding = re.search("|".join(embedding_names), results_file).group()
    # Use a regular expression to search in the file name for the number of qubits and layers
    # The regular expression searches for the string "wxdy", where x and y are integers
    # and returns x and y as groups
    num_qubits, num_layers = re.search(r"w(\d+)d(\d+)", results_file).groups()
    # If "trained" is in the file name, then search for a boolean value and append to the embedding name either "trained_False" or "trained_True"
    if "trained" in results_file:
        trained = re.search(r"trained_(\w+)", results_file).group(1)
        embedding = embedding + "_trained_" + trained

    return embedding, int(num_qubits), int(num_layers)


def running_average_filter(points, factor=0.3):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def get_classification_metrics(clf: SVC, X: np.ndarray, y_true: np.ndarray):
    """Returns accuracy, Jaccard index, precision, recall and specificity
    of the classifier.

    Args:
        clf (SVC): the SVC classifier
        X (np.ndarray): input samples
        y_true (np.ndarray): the true labels
    """
    y_pred = clf.predict(X)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tn + fp) / (tn + fp + fn + fp)
    jaccard = tp / (tp + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, jaccard, precision, recall, specificity
