import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV


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
