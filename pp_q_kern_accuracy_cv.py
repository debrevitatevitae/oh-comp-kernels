import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from project_directories import RESULTS_DIR, GRAPHICS_DIR

if __name__ == "__main__":
    start = time.time()

    # Load results into DataFrame
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    # [3:] removes the heading 'pp_'
    python_results_file_name = python_file_name_no_ext[3:]

    df = pd.read_csv(RESULTS_DIR / f"{python_results_file_name}.csv")

    # Plot results with seaborn
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.5)
    sns.set_palette("colorblind")
    sns.set_style("ticks")

    # Create subplots for each embedding
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 12), sharex=True, sharey=True)

    for i, embedding in enumerate(["he2w3d2", "he2w3d3", "he2w3d4", "he2w6d2", "he2w6d3", "he2w6d4"]):
        ax = axes[i // 3, i % 3]

        # Filter data for the current embedding
        embedding_data = df[df["embedding_name"] == embedding]

        # Plot for "random" data
        sns.lineplot(
            x="param_C",
            y="mean_test_score",
            hue="trained",
            style="trained",
            hue_order=[False, True],
            style_order=[False, True],
            markers=True,
            dashes=False,
            legend="brief",
            data=embedding_data,
            ax=ax
        )
        ax.set_xscale("log")
        ax.set_title(f"{embedding}")
        ax.set_xlabel("C")
        ax.set_ylabel("Test accuracy")

        # Remove titles in legends
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:], labels=["Random", "Trained"])

    plt.tight_layout()
    plt.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
