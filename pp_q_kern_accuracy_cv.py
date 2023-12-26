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
    sns.set_context("paper", font_scale=1.2)
    sns.set_palette("colorblind")
    sns.set_style("ticks")

    # Create subplots for each embedding
    fig, axes = plt.subplots(4, 3, figsize=(
        8.27, 11.69), sharex=True, sharey=True)
    fig.text(0.5, 0.01, "C", ha='center')
    fig.text(0.005, 0.5, "Mean validation accuracy",
             va='center', rotation='vertical')

    for i, embedding in enumerate(["iqpw3d1", "iqpw3d2", "iqpw3d3", "iqpw4d1", "iqpw4d2", "iqpw4d3", "iqpw5d1", "iqpw5d2", "iqpw5d3", "iqpw6d1", "iqpw6d2", "iqpw6d3"]):
        ax = axes[i // 3, i % 3]

        # Filter data for the current embedding
        embedding_data = df[df["embedding_name"] == embedding]

        # Plot for "random" data
        sns.lineplot(
            x="param_C",
            y="mean_test_score",
            markers=True,
            dashes=False,
            data=embedding_data,
            ax=ax
        )
        ax.set_xscale("log")
        ax.set_title(f"{embedding}")

        # keep only 5 yticks
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        # set limit for y axis
        ax.set_ylim([0.5, 1])

        # Reduce the number of decimal digits to 2
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{round(x, 2)}'))
        ax.set_xlabel("")
        ax.set_ylabel("")

        # reduce font size of x and y ticks
        ax.tick_params(axis='both', which='major', labelsize=10)

        # add an inset axis with a barplot of std_test_score
        inset_ax = ax.inset_axes([0.55, 0.05, 0.45, 0.45])
        sns.barplot(
            x="param_C",
            y="std_test_score",
            data=embedding_data,
            ax=inset_ax,
            width=0.6  # Adjust the width value as needed
        )

        # set limit for y axis
        inset_ax.set_ylim([0, 0.05])

        # remove labels and ticks from inset axis
        inset_ax.set_xlabel("")  # Set x label
        inset_ax.set_xticklabels([])
        inset_ax.set_ylabel("")  # Set y label

        # reduce yticks to 5
        inset_ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        # reduce font size of y ticks
        inset_ax.tick_params(axis='y', labelsize=8)

        # set title for inset axis
        inset_ax.set_title("Std val acc")

    plt.tight_layout()
    plt.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
