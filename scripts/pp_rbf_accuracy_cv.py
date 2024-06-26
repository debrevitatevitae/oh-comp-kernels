import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ohqk.project_directories import GRAPHICS_DIR, RESULTS_DIR

if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 12

    start = time.time()

    # Load results into DataFrame
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    # [3:] removes the heading 'pp_'
    python_results_file_name = python_file_name_no_ext[3:]

    df = pd.read_csv(
        RESULTS_DIR / f"{python_results_file_name}.csv",
        dtype={
            "param_C": float,
            "param_gamma": float,
            "mean_test_score": float,
            "std_test_score": float,
        },
    )

    # Create a pivot table for better visualization
    pivot_table = df.pivot(
        index="param_C", columns="param_gamma", values="mean_test_score"
    )

    # Create a heatmap of the pivot table with C and gamma on the axes, C in exponential notation
    plt.figure(figsize=(10, 10))

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Mean Vaidation Accuracy"},
        xticklabels=pivot_table.columns.round(3),
        yticklabels=[r"$10^{-1}$"]
        + [rf"$10^{i}$" for i in np.arange(0, 8, 1, dtype=int)],
    )

    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$C$")
    plt.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
