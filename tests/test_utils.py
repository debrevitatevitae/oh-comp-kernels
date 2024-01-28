import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ohqk.utils import get_info_from_results_file_name, load_split_scale_data


def test_load_split_scale_data(monkeypatch):
    # Generate dummy data
    data = {
        "eps11": [1, 2, 3, 4, 5],
        "eps22": [6, 7, 8, 9, 10],
        "eps12": [11, 12, 13, 14, 15],
        "failed": [0, 1, 0, 1, 0],
    }
    df_data = pd.DataFrame(data)

    # Set the random seed for reproducibility
    np.random.seed(42)

    # Generate reference splits
    X_train_ref, X_test_ref, y_train_ref, y_test_ref = train_test_split(
        df_data[["eps11", "eps22", "eps12"]], df_data["failed"], test_size=0.2
    )

    # Mock the necessary functions
    def mock_read_csv(filepath):
        return df_data

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)

    # Test the function with default arguments
    np.random.seed(42)  # seed must be set again!
    X_train, X_test, y_train, y_test = load_split_scale_data(
        test_size=0.2, scale="standard", to_jax=False
    )

    # Check the shapes of the train and test sets
    assert X_train.shape == X_train_ref.shape
    assert X_test.shape == X_test_ref.shape
    assert y_train.shape == y_train_ref.shape
    assert y_test.shape == y_test_ref.shape

    # Check the scaling method
    scaler = StandardScaler()
    X_train_ref_scaled = scaler.fit_transform(X_train_ref)
    assert np.allclose(X_train, X_train_ref_scaled)

    # Test the function with scale="angle"
    np.random.seed(42)  # seed must be set again!
    X_train, X_test, y_train, y_test = load_split_scale_data(
        test_size=0.2, scale="angle", to_jax=False
    )

    # Check the scaling method
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train_ref_scaled = scaler.fit_transform(X_train_ref)
    assert np.allclose(X_train, X_train_ref_scaled)

    # Test the function with to_jax=True
    X_train, X_test, y_train, y_test = load_split_scale_data(
        test_size=0.2, scale="standard", to_jax=True
    )

    # Check the data type conversion
    assert isinstance(X_train, jnp.ndarray)
    assert isinstance(y_train, jnp.ndarray)
    assert isinstance(X_test, jnp.ndarray)
    assert isinstance(y_test, jnp.ndarray)


def test_get_info_from_results_file_name():
    # Test case 1: results_file = "iqp_w2d3_results.txt"
    embedding, num_qubits, num_layers = get_info_from_results_file_name(
        "cv_iqpw2d3.csv"
    )
    assert embedding == "iqp"
    assert num_qubits == 2
    assert num_layers == 3

    # Test case 2: results_file = "he2_untrained_w4d5_results.txt"
    embedding, num_qubits, num_layers = get_info_from_results_file_name(
        "cv_he2w4d5_trained_False.csv"
    )
    assert embedding == "he2_trained_False"
    assert num_qubits == 4
    assert num_layers == 5

    # Test case 3: results_file = "he2_trained_w1d2_results.txt"
    embedding, num_qubits, num_layers = get_info_from_results_file_name(
        "cv_he2w1d2_trained_True.csv"
    )
    assert embedding == "he2_trained_True"
    assert num_qubits == 1
    assert num_layers == 2


if __name__ == "__main__":
    pytest.main()
