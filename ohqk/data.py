from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class LabelledDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


def read_file_to_numpy_array(filepath, dim_input=1, sample_size=None):
    """Reads a comma-separated file and returns a numpy array."""
    data = np.genfromtxt(
        filepath, dtype=np.float32, delimiter=",", skip_header=1
    )

    if sample_size is not None:
        data = data[
            np.random.choice(data.shape[0], sample_size, replace=False)
        ]

    dim_output = data.shape[1] - dim_input

    return data[:, :dim_input].reshape(-1, dim_input), data[
        :, dim_input:
    ].reshape(-1, dim_output)


def assign_labels(strains, stresses, curve_idx=0, residual_stiffness=0.9):
    """
    Assigns labels to data points based on the given strains and stresses.

    Parameters:
        strains (ndarray): Array of strains.
        stresses (ndarray): Array of stresses.
        curve_idx (int, optional): Index of the curve to consider. Defaults to 0.
        residual_stiffness (float, optional): Residual stiffness threshold. Defaults to 0.9.

    Returns:
        ndarray: Array of labels assigned to each data point.
    """

    N, _ = stresses.shape

    threshold = N - 1
    stiffness_init = stresses[0, curve_idx] / strains[0, curve_idx]
    for i in range(1, N):
        stiffness = stresses[i, curve_idx] / strains[i, curve_idx]
        if stiffness <= residual_stiffness * stiffness_init:
            threshold = i
            break

    labels = np.array(
        [-1 if i < threshold else 1 for i in range(N)], dtype=np.int32
    )
    return labels


class StrainsStressesLabels:
    """Class for loading and labeling strains and stresses"""

    def __init__(
        self, data_file_path: Path | str, labeling_threshold: float = None
    ):
        self.data_file_path = data_file_path

        self.strains, self.stresses = read_file_to_numpy_array(
            self.data_file_path,
            dim_input=3,
        )

        self.labels = (
            assign_labels(
                self.strains,
                self.stresses,
                residual_stiffness=labeling_threshold,
            )
            if labeling_threshold
            else np.zeros(len(self.strains))
        )

    def __len__(self):
        return len(self.strains)

    def __getitem__(self, index: int):
        x = self.strains[index]
        y = self.stresses[index]
        z = self.labels[index]
        return x, y, z
