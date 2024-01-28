import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ohqk.project_directories import GRAPHICS_DIR, RESULTS_DIR

if __name__ == "__main__":
    start = time.time()

    # Load rbf results into DataFrame
    rbf_results_file_name = "rbf_accuracy_test"
    rbf_df = pd.read_csv(RESULTS_DIR / f"{rbf_results_file_name}.csv")

    # Load results into DataFrame
    q_kerns_names = ["he2w3d2", "he2w6d4"]
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    # [3:] removes the heading 'pp_'
    python_results_file_name = python_file_name_no_ext[3:]

    q_df_1 = pd.read_csv(
        RESULTS_DIR / f"{python_results_file_name}_{q_kerns_names[0]}.csv"
    )
    q_df_2 = pd.read_csv(
        RESULTS_DIR / f"{python_results_file_name}_{q_kerns_names[1]}.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        rbf_df["train_size"],
        rbf_df["mean_test_accuracy"],
        yerr=rbf_df["std_test_accuracy"],
        fmt="o",
        capsize=4,
    )
    ax.errorbar(
        q_df_1["train_size"],
        q_df_1["mean_test_accuracy"],
        yerr=q_df_1["std_test_accuracy"],
        fmt="o",
        capsize=4,
    )
    ax.errorbar(
        q_df_2["train_size"],
        q_df_2["mean_test_accuracy"],
        yerr=q_df_2["std_test_accuracy"],
        fmt="o",
        capsize=4,
    )
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Test accuracy")
    ax.legend(["RBF", "w3d2", "w6d4"])

    plt.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
