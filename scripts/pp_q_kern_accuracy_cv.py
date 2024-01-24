import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ohqk.project_directories import RESULTS_DIR, GRAPHICS_DIR


if __name__ == "__main__":
    start = time.time()

    # Load all results files that contain `iqp` in their name
    results_files = [
        f for f in os.listdir(RESULTS_DIR)
        if "iqp" in f and f.endswith(".csv")
    ]

    # order the results first by the number following 'w' and then by the number following 'd'

    # Define a color palette in seaborn, made of shades of the same color (red), one for each result file
    # This will be used to color the different scatterplot points
    palette = sns.color_palette("Reds", len(results_files))

    # Loop over the results files and plot the mean test scores in a seaborn scatterplot
    for i, results_file in enumerate(results_files):
        df = pd.read_csv(RESULTS_DIR / results_file)
        # Get the architecture name from the file name
        architecture = results_file.split("_")[4]
        ax = sns.scatterplot(
            data=df,
            x="param_C",
            y="mean_test_score",
            color=palette[i],
            label=architecture,
        )
        ax.set(xscale="log")  # Set x-axis to logarithmic scale

    # in the legend, only show the one point and name it "iqp"
    plt.legend(
        handles=ax.legend_.legendHandles[:1],
        labels=["iqp"],
        # Adjust the y-coordinate to move the legend above the plot
        bbox_to_anchor=(0.5, 1.15),
        loc="upper center",
        borderaxespad=0.,
    )

    # Define the results file
    python_results_file_name = os.path.basename(__file__)
    python_results_file_name_no_ext = os.path.splitext(
        python_results_file_name)[0]

    plt.savefig(GRAPHICS_DIR / f"{python_results_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
