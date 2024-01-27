import pytest

from ohqk.utils import get_info_from_results_file_name


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
