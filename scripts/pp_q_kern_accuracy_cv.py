import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ohqk.project_directories import RESULTS_DIR, GRAPHICS_DIR


if __name__ == "__main__":
    start = time.time()

    # Load the results files for the 3 different embeddings
    results_iqp_files = [
        f for f in os.listdir(RESULTS_DIR)
        if "iqp" in f and f.endswith(".csv")
    ]
    results_he2_untrained_files = [
        f for f in os.listdir(RESULTS_DIR)
        if "he2" in f and "trained_False" in f and f.endswith(".csv")
    ]
    results_he2_trained_files = [
        f for f in os.listdir(RESULTS_DIR)
        if "he2" in f and "trained_True" in f and f.endswith(".csv")
    ]

    # Concatenate the results files into one list
    results_files = results_iqp_files + \
        results_he2_untrained_files + results_he2_trained_files

    # Define a color palette in seaborn, made of shades of the same color for each embedding
    # This will be used to color the different scatterplot points
    palette = sns.color_palette("Reds", len(results_iqp_files)) + \
        sns.color_palette("Blues", len(results_he2_untrained_files)) + \
        sns.color_palette("Greens", len(results_he2_trained_files))

    # Creare a figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Loop over the results files and plot the mean test scores in a seaborn scatterplot
    for i, results_file in enumerate(results_files):
        df = pd.read_csv(RESULTS_DIR / results_file)
        # Get the architecture name from the file name
        # architecture = results_file.split("_")[4]
        sns.scatterplot(
            data=df,
            x="param_C",
            y="mean_test_score",
            color=palette[i],
            ax=ax,
            # label=architecture,
        )
        ax.set(xscale="log")  # Set x-axis to logarithmic scale

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
        python_results_file_name)[0]

    plt.savefig(GRAPHICS_DIR / f"{python_results_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
