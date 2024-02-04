import csv
import os
import random
import re

import numpy as np

from ohqk.data import read_file_to_numpy_array
from ohqk.project_directories import PROC_DATA_DIR, RAW_DATA_DIR


def get_file_names(dir=RAW_DATA_DIR, shuffle=True):
    """
    Retrieve a list of file names from the specified directory.

    Parameters
    ----------
    dir : str, optional
        The directory from which to retrieve file names. Default is 'RAW_DATA_DIR'.
    shuffle : bool, optional
        Whether to shuffle the list of file names. Default is True.

    Returns
    -------
    list of str
        A list containing file names from the specified directory.

    Examples
    --------
    >>> get_file_names()
    ['ExampleFile1.txt', 'ExampleFile2.txt', ...]

    >>> get_file_names(dir='custom_directory', shuffle=False)
    ['File1.txt', 'File2.txt', ...]

    Notes
    -----
    The function retrieves file names from the specified directory. If the `shuffle` parameter
    is set to True (default), the order of the file names in the returned list will be randomized.

    """
    fnames = [f for f in os.listdir(dir) if f[:2] == "Ea"]
    if shuffle:
        random.shuffle(fnames)
    return fnames


def sample_and_label(
    strains, stresses, max_loads, residual_stiffness=0.9, N_samples=10
):
    """
    Randomly sample and label strains for binary classification based on loading curves.

    Parameters
    ----------
    strains : ndarray
        2D array representing strains, where each row corresponds to a load increment.
    stresses : ndarray
        2D array representing stresses, where each row corresponds to a load increment.
    max_loads : ndarray
        1D array representing the maximum loads for each loading curve.
    residual_stiffness : float, optional
        The threshold for residual stiffness to determine the drop in stiffness. Default is 0.9.
    N_samples : int, optional
        Number of samples to generate for each class. Default is 10.

    Returns
    -------
    tuple of ndarray
        A tuple containing two NumPy arrays:
        - Strains samples with shape (2 * N_samples, num_features).
        - Labels with shape (2 * N_samples,), where labels are -1 or 1.

    Examples
    --------
    >>> strains, stresses, max_loads = load_data()
    >>> sample_and_label(strains, stresses, max_loads, residual_stiffness=0.8, N_samples=5)
    (array([[...], [...], ...]),
     array([-1, -1, ..., 1, 1]))

    Notes
    -----
    This function randomly samples strains for binary classification based on loading curves.
    It selects the loading curve with the maximum final load and determines a threshold
    based on a drop in stiffness, controlled by the `residual_stiffness` parameter.
    Strains are then randomly sampled from class '-1' and '+1' based on whether they occur
    before or after the threshold. Labels are assigned accordingly.

    """
    N, _ = stresses.shape

    curve_idx = np.argmax(np.abs(max_loads))

    threshold = N - 1
    stiffness_init = stresses[0, curve_idx] / strains[0, curve_idx]
    for i in range(1, N):
        stiffness = stresses[i, curve_idx] / strains[i, curve_idx]
        if stiffness <= residual_stiffness * stiffness_init:
            threshold = i
            break

    idxs_class_1 = np.random.randint(0, threshold, N_samples)
    idxs_class_2 = np.random.randint(threshold, N, N_samples)

    strains_class_1 = strains[idxs_class_1]
    strains_class_2 = strains[idxs_class_2]
    strains_samples = np.append(strains_class_1, strains_class_2, axis=0)
    labels = [-1 if i < N_samples else 1 for i in range(2 * N_samples)]
    labels = np.array(labels, dtype=np.int32)

    return strains_samples, labels


if __name__ == "__main__":
    # define a seed and seed the generators for reproducibility
    seed = 42
    random.seed(seed)

    # Generate data
    files = get_file_names()

    # take n samples from each file, label them and write to a csv file
    with open(PROC_DATA_DIR / "data_labeled.csv", "w") as ff:
        writer = csv.writer(ff, delimiter=",")
        writer.writerow(["eps11", "eps22", "eps12", "failed"])  # csv title row

        for f in files:
            max_loads = re.findall("-?[0-9]+", f)  # loads in the file name
            max_loads = [float(l) for l in max_loads]
            strains, stresses = read_file_to_numpy_array(
                RAW_DATA_DIR / f, dim_input=3
            )
            x, y = sample_and_label(strains, stresses, max_loads)

            # write to csv in the format: [xx[0], xx[1], xx[2]], y
            for xx, yy in zip(x, y):
                writer.writerow([*xx, yy])
