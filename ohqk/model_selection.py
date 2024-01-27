import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from ohqk.project_directories import RESULTS_DIR
from ohqk.utils import get_info_from_results_file_name


def run_grid_search_cv(
    X, y, estimator, param_grid, cv=5, n_jobs=-3, verbose=3, error_score=np.nan
):
    """Run a grid search cross validation and return the results as a dataframe. The output dataframe is similar to the output of `sklearn.model_selection.GridSearchCV.cv_results_`, but ignores NaNs in comuting the mean and standard deviation of the test scores, unless the majority of the test scores are NaNs, in which case the mean and standard deviation of the test scores are set to NaN."""
    grid_search = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        error_score=error_score,
    )

    grid_search.fit(X, y)

    df_results = pd.DataFrame(grid_search.cv_results_)

    # Create a new column with the indices of the splits which resulted into a NaN test score, for every row
    # Initialize 'nan_splits' as a DataFrame of lists
    df_results["nan_splits"] = df_results["split0_test_score"].apply(
        lambda x: [np.isnan(x)]
    )

    # Append to the lists in the loop
    for i in range(1, cv):
        df_results["nan_splits"] = df_results["nan_splits"] + df_results[
            f"split{i}_test_score"
        ].apply(lambda x: [np.isnan(x)])

    # If the majority of the test scores are NaNs, then the mean and standard deviation of the test scores are set to NaN
    df_results["mean_test_score"] = df_results.apply(
        lambda row: np.nan
        if np.sum(row["nan_splits"]) >= cv / 2
        else row["mean_test_score"],
        axis=1,
    )

    df_results["std_test_score"] = df_results.apply(
        lambda row: np.nan
        if np.sum(row["nan_splits"]) >= cv / 2
        else row["std_test_score"],
        axis=1,
    )

    return df_results


def find_best_scoring_embeddings(
    results_files,
    embedding_names=["iqp", "he2_trained_False", "he2_trained_True"],
    save_to_csv=False,
):
    """Searches the results files for the best scoring case for each embedding
    name. Returns a dataframe and optionally saves it to a csv file."""
    # Initialize a list of dictionaries, which will be used to create a dataframe
    # Each dictionary will contain the embedding name, the number of qubits and
    # the number of layers of the best scoring case for each embedding
    best_scoring_embeddings = []

    for embedding_name in embedding_names:
        # Initialize the best score and the best scoring case for each embedding
        best_score = 0
        best_rf = None
        best_param_C = None

        for results_file in results_files:
            (
                embedding,
                _,
                _,
            ) = get_info_from_results_file_name(results_file)

            # If the embedding name matches the current embedding name in the loop
            if embedding == embedding_name:
                df = pd.read_csv(RESULTS_DIR / results_file)

                # Find the best scoring case for the current embedding
                best_score_i = df["mean_test_score"].max()

                # If the best scoring case for the current embedding is better
                # than the best scoring case so far, update the best score and
                # the best scoring case
                if best_score_i > best_score:
                    best_score = best_score_i
                    best_rf = results_file
                    best_param_C = df.loc[df["mean_test_score"].idxmax()][
                        "param_C"
                    ]

        # Append the best scoring case for the current embedding to the list
        _, num_qubits, num_layers = get_info_from_results_file_name(best_rf)

        best_scoring_embeddings.append(
            {
                "embedding": embedding_name,
                "num_qubits": num_qubits,
                "num_layers": num_layers,
                "best_score": best_score,
                "param_C": best_param_C,
                "var_params_filename": f"q_kern_kta_opt_he2w{num_qubits}d{num_layers}.pkl"
                if embedding_name == "he2_trained_True"
                else None,
            }
        )

    # Create a dataframe from the list of dictionaries
    df_best_scoring_embeddings = pd.DataFrame(best_scoring_embeddings)

    if save_to_csv:
        df_best_scoring_embeddings.to_csv(
            RESULTS_DIR / "best_scoring_embeddings.csv", index=False
        )

    return df_best_scoring_embeddings
