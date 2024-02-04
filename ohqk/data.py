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


def read_file_to_numpy_array(filepath, dim_input=1):
    """
    Read data from a file into a NumPy array and reshape it.

    Parameters
    ----------
    filepath : str
        The path to the CSV file containing the data.
    dim_input : int, optional
        The number of columns to consider as input features. Default is 1.

    Returns
    -------
    tuple of ndarray
        A tuple containing two NumPy arrays:
        - The input data with shape (-1, dim_input).
        - The labels (output) data with shape (-1, dim_output), where dim_output
        is calculated as the total number of columns minus dim_input.

    Examples
    --------
    >>> read_file_to_numpy_array('data.csv')
    (array([[1.0],
            [2.0],
            ...]),
     array([[0.5],
            [1.5],
            ...]))

    >>> read_file_to_numpy_array('data.csv', dim_input=2)
    (array([[1.0, 2.0],
            [3.0, 4.0],
            ...]),
     array([[0.5],
            [1.5],
            ...]))

    Notes
    -----
    This function reads data from a CSV file using NumPy's `genfromtxt` function.
    It assumes that the input features are located in the first `dim_input` columns,
    and the remaining columns are considered as output labels.
    The resulting arrays are reshaped for compatibility with machine learning frameworks.

    """
    data = np.genfromtxt(
        filepath, dtype=np.float32, delimiter=",", skip_header=1
    )

    dim_output = data.shape[1] - dim_input

    return data[:, :dim_input].reshape(-1, dim_input), data[
        :, dim_input:
    ].reshape(-1, dim_output)
