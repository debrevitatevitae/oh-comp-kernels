import os
import time
from matplotlib import pyplot as plt

import pandas as pd

from project_directories import RESULTS_DIR, GRAPHICS_DIR


if __name__ == '__main__':
    start = time.time()

    # Load results into DataFrame
    python_file_name = os.path.basename(__file__)
    python_file_name_no_ext = os.path.splitext(python_file_name)[0]
    # [3:] removes the heading 'pp_'
    python_results_file_name = python_file_name_no_ext[3:]

    df = pd.read_csv(RESULTS_DIR / f"{python_results_file_name}.csv")

    # Set up the figure and axes
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    # Flatten the 2D array of axes for easier iteration
    axes = axes.flatten()

    # Plot each column
    for i, column in enumerate(df.columns):
        ax = axes[i]
        ax.hist(df[column], bins=5, color='skyblue', edgecolor='black')
        ax.set_title(column)
        ax.set_xlabel('KTA')
        ax.set_ylabel('Frequency')

    # Adjust layout to prevent overlapping titles
    fig.tight_layout()

    # Show the plot
    fig.savefig(GRAPHICS_DIR / f"{python_file_name_no_ext}.pdf")

    exec_time = time.time() - start
    minutes = int(exec_time // 60)
    seconds = int(exec_time % 60)

    print(f"Script execution time: {minutes} minutes and {seconds} seconds")
