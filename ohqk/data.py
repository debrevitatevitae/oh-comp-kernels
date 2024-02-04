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
