import pandas as pd
import pytest

from ohqk.model_selection import find_best_scoring_embeddings
from ohqk.project_directories import RESULTS_DIR


def test_find_best_scoring_embeddings(monkeypatch):
    # Mock results files
    results_files = [
        "cv_iqpw2d3.csv",
        "cv_he2w4d5_trained_False.csv",
        "cv_he2w1d2_trained_True.csv",
    ]

    # Mock dataframes
    df_iqp = pd.DataFrame(
        {"mean_test_score": [0.8, 0.6, 0.7], "param_C": [1.0, 2.0, 3.0]}
    )
    df_he2_untrained = pd.DataFrame(
        {"mean_test_score": [0.9, 0.5, 0.4], "param_C": [4.0, 5.0, 6.0]}
    )
    df_he2_trained = pd.DataFrame(
        {"mean_test_score": [0.7, 0.8, 0.9], "param_C": [7.0, 8.0, 9.0]}
    )

    # Mock get_info_from_results_file_name function
    def mock_get_info_from_results_file_name(results_file):
        if results_file == "cv_iqpw2d3.csv":
            return "iqp", 2, 3
        elif results_file == "cv_he2w4d5_trained_False.csv":
            return "he2_trained_False", 4, 5
        elif results_file == "cv_he2w1d2_trained_True.csv":
            return "he2_trained_True", 1, 2

    # Mock pd.read_csv function
    def mock_read_csv(filepath):
        if filepath == RESULTS_DIR / "cv_iqpw2d3.csv":
            return df_iqp
        elif filepath == RESULTS_DIR / "cv_he2w4d5_trained_False.csv":
            return df_he2_untrained
        elif filepath == RESULTS_DIR / "cv_he2w1d2_trained_True.csv":
            return df_he2_trained

    # Mock the necessary functions
    monkeypatch.setattr(
        "ohqk.model_selection.get_info_from_results_file_name",
        mock_get_info_from_results_file_name,
    )
    monkeypatch.setattr("pandas.read_csv", mock_read_csv)

    # Call the function
    df_best_scoring_embeddings = find_best_scoring_embeddings(results_files)

    # Assert the expected output
    expected_output = pd.DataFrame(
        {
            "embedding": ["iqp", "he2_trained_False", "he2_trained_True"],
            "num_qubits": [2, 4, 1],
            "num_layers": [3, 5, 2],
            "best_score": [0.8, 0.9, 0.9],
            "param_C": [1.0, 4.0, 9.0],
            "var_params_filename": [
                None,
                None,
                "q_kern_kta_opt_he2w1d2.pkl",
            ],
        }
    )
    pd.testing.assert_frame_equal(df_best_scoring_embeddings, expected_output)


if __name__ == "__main__":
    pytest.main()
