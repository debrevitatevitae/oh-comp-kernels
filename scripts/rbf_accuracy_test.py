import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ohqk.project_directories import RESULTS_DIR
from ohqk.utils import load_split_scale_data

if __name__ == "__main__":
    start = time.time()

    np.random.seed(42)

    # data loading, splitting, and scaling
    X_train_scaled, X_test_scaled, y_train, y_test = load_split_scale_data(
        test_size=0.2, scale="standard"
    )

    N = len(y_train)

    # get the optimal (C, \gamma) from the cross validation results file
    df_cv_results = pd.read_csv(RESULTS_DIR / "rbf_accuracy_cv.csv")
    idx_opt = df_cv_results["mean_test_score"].argmax()
    C_opt = df_cv_results["param_C"][idx_opt]
    gamma_opt = df_cv_results["param_gamma"][idx_opt]

    print(f"Optimal C: {C_opt}")
    print(f"Optimal gamma: {gamma_opt}")

    Cs = [1e4, C_opt]

    # create a DataFrame for the results, which includes the training set size, mean test accuracy, the standard deviation of the test accuracy and the number of support vectors
    out_df = pd.DataFrame(
        columns=[
            "kernel",
            "train_size",
            "mean_test_accuracy",
            "std_test_accuracy",
            "n_support_vectors",
        ]
    )

    # for each of the Cs, repeat a training and compute the test accuracy
    for C in Cs:
        # declare some training set sizes
        train_sizes = [int(N * frac) for frac in np.arange(0.1, 1.1, 0.1)]

        # for each of the train sizes, repeat a training and compute the test accuracy
        num_reps = 10
        for i, ts in enumerate(train_sizes):
            accuracies = []
            for _ in range(num_reps):
                idxs_selection = np.random.choice(N, size=ts)
                X_train_scaled_selection = X_train_scaled[idxs_selection]
                y_train_selection = y_train[idxs_selection]
                # create SVC with optimal hyperparameters
                clf = SVC(kernel="rbf", gamma=gamma_opt, C=C)
                # fit the classifier
                clf.fit(X_train_scaled_selection, y_train_selection)
                accuracies.append(clf.score(X_test_scaled, y_test))

            out_df.loc[len(out_df)] = {
                "kernel": f"rbf_C{C:.1E}",
                "train_size": ts,
                "mean_test_accuracy": np.mean(accuracies),
                "std_test_accuracy": np.std(accuracies),
                "n_support_vectors": clf.n_support_.sum(),
            }

    # save DataFrame to a csv file named after this Python file
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    out_df.to_csv(RESULTS_DIR / f"{python_file_name_no_ext}.csv", index=False)

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
