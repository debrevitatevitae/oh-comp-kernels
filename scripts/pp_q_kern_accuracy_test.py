import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ohqk.project_directories import GRAPHICS_DIR, RESULTS_DIR

if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 12

    start = time.time()

    # Load test accuracy results into DataFrame and merge into one
    rbf_results_file_name = "rbf_accuracy_test"
    q_results_file_name = "q_kern_accuracy_test"
    rbf_df = pd.read_csv(RESULTS_DIR / f"{rbf_results_file_name}.csv")
    q_df = pd.read_csv(RESULTS_DIR / f"{q_results_file_name}.csv")
    # merge rbf_df and q_df
    results_df = pd.concat([rbf_df, q_df])

    # Make a barplot
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    # sns.set_context("paper")
    sns.set_theme(font_scale=1.2)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.grid(color="grey")
    ax.set_facecolor("white")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Mean Test Accuracy")
    ax.set_ylim(0.5, 0.95)
    ax = sns.barplot(
        x="train_size",
        y="mean_test_accuracy",
        hue="kernel",
        data=results_df,
        ax=ax,
        legend="auto",
    )
    # following, hack to get the handles for later legend editing
    patches = [
        matplotlib.patches.Patch(color=sns.color_palette()[i], label=t)
        for i, t in enumerate(t.get_text() for t in ax.get_xticklabels())
    ]

    # Move the legend above the plot and reduce the font size.
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=5,
        fontsize=12,
        facecolor="white",
        labels=[
            "RBF, C=1e4",
            "RBF, C=1e7",
            "IQP",
            "HE2, random params",
            "HE2, trained",
        ],
    )

    # Output file
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]

    plt.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
