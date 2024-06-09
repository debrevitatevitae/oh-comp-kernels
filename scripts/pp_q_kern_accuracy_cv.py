import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ohqk.model_selection import find_best_scoring_embeddings
from ohqk.project_directories import GRAPHICS_DIR, RESULTS_DIR
from ohqk.utils import find_order_concatenate_cv_result_files

if __name__ == "__main__":
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 12

    start = time.time()

    results_files = find_order_concatenate_cv_result_files()

    embedding_types = ["iqp", "he2_trained_False", "he2_trained_True"]
    embedding_names = ["IQP", "HE2, random parameters", "HE2, trained"]
    num_architectures = len(results_files) // 3

    # Find the best scoring embeddings for each embedding type
    best_embeddings = find_best_scoring_embeddings(
        results_files, embedding_types, save_to_csv=True
    )

    # Define a color palette in seaborn, made of shades of the same color for each embedding
    # This will be used to color the different scatterplot points
    palette = (
        sns.color_palette("Reds", num_architectures)
        + sns.color_palette("Blues", num_architectures)
        + sns.color_palette("Greens", num_architectures)
    )

    # Creare a figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(10, 8), sharey=True)

    # Loop over the results files and plot the mean test scores in a seaborn scatterplot
    for i, results_file in enumerate(results_files):
        df = pd.read_csv(RESULTS_DIR / results_file)
        # Get the architecture name from the file name
        # architecture = results_file.split("_")[4]
        ax = axs[i // num_architectures]
        sns.scatterplot(
            data=df, x="param_C", y="mean_test_score", color=palette[i], ax=ax
        )

    for i in range(3):
        # Set x-axis to logarithmic scale
        axs[i].set(
            xscale="log",
            title=embedding_names[i],
            ylim=(0.6, 0.9),
            xlabel="$C$",
            ylabel="Mean Test Accuracy",
        )
        axs[i].set_xticks(
            [0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0],
            labels=[
                r"$10^{-1}$",
                r"$10^{0}$",
                r"$10^{1}$",
                r"$10^{2}$",
                r"$10^{3}$",
                r"$10^{4}$",
                r"$10^{5}$",
            ],
        )

    fig.tight_layout()

    # plt.legend(
    #     # the legend handles correspond to the first architecture of each embedding
    #     handles=[ax.legend_.legendHandles[0], ax.legend_.legendHandles[len(
    #         results_iqp_files)], ax.legend_.legendHandles[
    #             len(results_iqp_files) + len(results_he2_untrained_files)
    #     ],
    #     ],
    #     labels=["iqp", "he2_untrained", "he2_trained"],
    #     # Adjust the y-coordinate to move the legend above the plot
    #     bbox_to_anchor=(0.5, 1.15),
    #     loc="upper center",
    #     borderaxespad=0.,
    # )

    # Define the results file
    python_results_file_name = os.path.basename(__file__)
    python_results_file_name_no_ext = os.path.splitext(
        python_results_file_name
    )[0]

    plt.savefig(GRAPHICS_DIR / f"{python_results_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
